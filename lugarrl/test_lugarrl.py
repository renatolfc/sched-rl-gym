#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest

from . import lugarrl, simulator, job, workload, fifo_scheduler, resource_pool, event, heap, scheduler


class MockLugarRL(object):
    def __getattr__(self, item):
        return lambda: None


class TestLugarRL(unittest.TestCase):
    def setUp(self):
        self.lugar = MockLugarRL()

    def test_single_step(self):
        self.lugar.step()


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        self.workload = workload.BinomialWorkloadGenerator(
            0.7, 0.8, self.small_job_parameters, self.large_job_parameters, length=1
        )
        self.scheduler = fifo_scheduler.FifoScheduler(16, 64)

    def test_time_based_simulator(self):
        sim = simulator.Simulator.make(simulator.SimulationType.TIME_BASED,
                                       self.workload,
                                       self.scheduler)
        self.assertEqual(0, sim.current_time)
        for i in range(100):
            sim.step()
        self.assertEqual(100, sim.current_time)

    def test_all_submission_times_are_different(self):
        sim = simulator.Simulator.make(simulator.SimulationType.TIME_BASED,
                                       self.workload,
                                       self.scheduler)
        for i in range(100):
            sim.step()
        prev = -1
        for j in sim.scheduler.all_jobs:
            self.assertNotEqual(j.submission_time, prev)
            prev = j.submission_time

    def test_not_all_jobs_executed(self):
        sim = simulator.Simulator.make(simulator.SimulationType.TIME_BASED,
                                       self.workload,
                                       self.scheduler)
        for i in range(100):
                sim.step()
        self.assertNotEqual(len(self.scheduler.queue_completed), len(self.scheduler.all_jobs))

    def test_all_jobs_executed(self):
        sim = simulator.Simulator.make(simulator.SimulationType.TIME_BASED,
                                       self.workload,
                                       self.scheduler)
        for i in range(100):
            sim.step()
        self.assertNotEqual(len(self.scheduler.queue_completed), len(self.scheduler.all_jobs))
        for i in range(1000):
            sim.step(False)
        self.assertEqual(len(self.scheduler.queue_completed), len(self.scheduler.all_jobs))

    def test_invalid_simulator(self):
        with self.assertRaises(RuntimeError):
            simulator.Simulator.make(
                simulator.SimulationType.EVENT_BASED,
                self.workload,
                self.scheduler
            )
        with self.assertRaises(RuntimeError):
            simulator.Simulator.make(
                42,
                self.workload,
                self.scheduler
            )


class TestTimeBaseLugarRL(TestLugarRL):
    def setUp(self):
        self.lugar = lugarrl.LugarRL()

    def test_time_based_simulator_instance(self):
        self.assertTrue(isinstance(self.lugar.simulator, simulator.TimeBasedSimulator))

    def test_step(self):
        self.assertEqual(self.lugar.simulator.current_time, 0)
        self.lugar.step()
        self.assertEqual(self.lugar.simulator.current_time, 1)

    def test_simulation_runs_fine(self):
        for i in range(1000):
            self.lugar.step()
        self.assertEqual(self.lugar.simulator.current_time, 1000)


class TestJobParameters(unittest.TestCase):
    def setUp(self):
        self.jp = job.JobParameters(1, 2, 1, 2, 1, 2)

    def test_first_job_id_isnt_zero(self):
        j = self.jp.sample()
        self.assertNotEqual(0, j.id)

    def test_any_negative_bounds_should_fail(self):
        with self.assertRaises(AssertionError):
            job.JobParameters(0, 0, 0, 0, 0, 0)

    def test_unable_to_go_back_in_time(self):
        with self.assertRaises(AssertionError):
            self.jp.add_time(-1)

    def test_time_increase(self):
        current_time = self.jp.time_step
        self.jp.add_time(1)
        self.assertEqual(current_time + 1, self.jp.time_step)
        self.jp.add_time()
        self.assertEqual(current_time + 2, self.jp.time_step)

    def test_job_ids_should_increase(self):
        last = self.jp.sample()
        for i in range(10):
            current = self.jp.sample()
            self.assertGreater(current.id, last.id)
            last = current

    def test_submission_time_should_equal_parameter(self):
        for i in range(10):
            time = random.randint(10, 10000)
            current_job = self.jp.sample(time)
            self.assertEqual(time, current_job.submission_time)

    def test_submission_time_different_from_parameter(self):
        self.jp.add_time()
        current_job = self.jp.sample(0)
        self.assertNotEqual(0, current_job.submission_time)


class MockScheduler(scheduler.Scheduler):
    def schedule(self):
        """"""


class TestScheduler(unittest.TestCase):
    events: event.EventQueue[event.JobEvent]

    @staticmethod
    def build_event(type, j, interval, time=0):
        p = resource_pool.IntervalTree([resource_pool.Interval(interval[0], interval[1])])
        j.processors_used = p
        return event.JobEvent(
            time, type, j
        )

    def new_job(self, processors, time):
        j = self.jp.sample()
        j.requested_processors = processors
        j.requested_time = time
        return j

    def build_event_pair(self, time, interval, j):
        return (
            self.build_event(event.EventType.JOB_START, j, interval, time),
            self.build_event(event.EventType.JOB_FINISH, j, interval, time + j.requested_time)
        )

    def setUp(self):
        self.scheduler = MockScheduler(10, 10000)
        self.jp = job.JobParameters(1, 2, 1, 2, 1, 2)
        self.events = event.EventQueue()

    def test_fits_empty_pool_without_events(self):
        j = self.jp.sample()
        self.assertTrue(self.scheduler.fits(0, j, self.scheduler.processor_pool.clone(), self.events)[0])
        j.requested_processors = 10
        self.assertTrue(self.scheduler.fits(0, j, self.scheduler.processor_pool.clone(), self.events)[0])

    def test_fits_empty_pool_with_events_in_the_future(self):
        j = self.jp.sample()
        j.requested_processors = 10
        j.requested_time = 5
        alloc, free = self.build_event_pair(5, (0, 10), j)
        self.events.add(alloc)
        self.events.add(free)

        j = self.jp.sample()
        j.requested_processors = 4
        j.requested_time = 5

        self.assertTrue(self.scheduler.fits(0, j, self.scheduler.processor_pool.clone(), self.events))

    def test_fits_partially_filled_pool_with_no_events(self):
        self.scheduler.processor_pool.allocate(resource_pool.IntervalTree([resource_pool.Interval(0, 6)]))
        j = self.jp.sample()
        self.assertTrue(self.scheduler.fits(0, j, self.scheduler.processor_pool.clone(), self.events))

    def test_doesnt_fit_fully_filled_pool_with_no_events(self):
        self.scheduler.processor_pool.allocate(resource_pool.IntervalTree([resource_pool.Interval(0, 10)]))
        j = self.jp.sample()
        self.assertFalse(self.scheduler.fits(0, j, self.scheduler.processor_pool.clone(), self.events))

    def test_past_events_dont_influence_the_present(self):
        j = self.jp.sample()
        j.requested_processors = 10
        j.requested_time = 5
        alloc, free = self.build_event_pair(0, (0, 10), j)
        self.events.add(alloc)
        self.events.add(free)

        j = self.jp.sample()
        j.requested_processors = 4
        j.requested_time = 5

        self.assertTrue(self.scheduler.fits(20, j, self.scheduler.processor_pool.clone(), self.events))

    def play_events(self, time):
        for e in (e for e in self.events if e.time <= time):
            if e.event_type == event.EventType.JOB_START:
                self.scheduler.processor_pool.allocate(e.job.processors_used)
            else:
                self.scheduler.processor_pool.deallocate(e.job.processors_used)

    def test_eventually_fits_partially_filled_pool(self):
        for i in range(5):
            alloc, free = self.build_event_pair(i, (i * 2, (i + 1) * 2), self.new_job(2, 6))
            self.events.add(alloc)
            self.events.add(free)
        j = self.new_job(2, 3)

        self.play_events(5)
        self.assertFalse(self.scheduler.fits(5, j, self.scheduler.processor_pool.clone(), self.events))

        self.scheduler = MockScheduler(10, 10000)
        self.play_events(6)
        self.assertTrue(self.scheduler.fits(6, j, self.scheduler.processor_pool.clone(), self.events))

    def test_should_fail_to_add_malformed_job(self):
        j = self.new_job(1, 0)
        with self.assertRaises(AssertionError):
            self.scheduler.add_job_events(j, 0)

    def test_should_fail_to_play_unsupported_event_type(self):
        j = self.jp.sample()
        j.requested_processors = 10
        j.requested_time = 5

        alloc, free = self.build_event_pair(5, (0, 10), j)
        alloc.event_type = event.EventType.RESOURCE_ALLOCATE

        with self.assertRaises(RuntimeError):
            self.scheduler.play_events([alloc, free], self.scheduler.processor_pool)

    def test_should_fail_to_find_resources_on_empty_cluster_with_large_job(self):
        self.scheduler = MockScheduler(16, 10000)

        j = self.new_job(17, 0)
        self.assertFalse(self.scheduler.processor_pool.find(
            j.requested_processors
        ))

        alloc, free = self.build_event_pair(0, (0, 17), j)
        self.assertFalse(self.scheduler.fits(
            0, j, self.scheduler.processor_pool, [alloc, free]
        ))


class TestFifoScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = fifo_scheduler.FifoScheduler(16, 64)
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        self.workload = workload.BinomialWorkloadGenerator(
            0.7, 0.8, self.small_job_parameters, self.large_job_parameters, length=1
        )
        self.counter = 0

    def make_job(self, submission, duration, processors):
        self.counter += 1
        return job.Job(self.counter, submission, duration, processors, 1, 1, processors, duration, 1,
                       job.JobStatus.SCHEDULED, 1, 1, 1, 1, 1, -1, -1, 0)

    def assertQueuesSane(self, time, completed, running, waiting, admission):
        self.assertEqual(self.scheduler.current_time, time)
        self.assertEqual(len(self.scheduler.queue_completed), completed)
        self.assertEqual(len(self.scheduler.queue_running), running)
        self.assertEqual(len(self.scheduler.queue_waiting), waiting)
        self.assertEqual(len(self.scheduler.queue_admission), admission)

    def test_all_jobs_completed(self):
        for i in range(100):
            self.scheduler.step()
            j = self.workload.sample(i)
            if j:
                self.scheduler.submit(j)
        for i in range(max([j.submission_time for j in self.scheduler.all_jobs]) + 1000):
            self.scheduler.step()
        self.scheduler.step()
        self.assertEqual(len(self.scheduler.queue_waiting), 0)
        for j in self.scheduler.all_jobs:
            self.assertTrue(j.status, job.JobStatus.COMPLETED)

    def test_single_job_executes_to_completion(self):
        j = self.small_job_parameters.sample(1)
        j.execution_time = 2
        self.scheduler.submit(j)
        self.assertQueuesSane(0, 0, 0, 0, 1)
        self.scheduler.step()
        self.assertQueuesSane(1, 0, 1, 0, 0)
        self.scheduler.step(2)
        self.assertQueuesSane(3, 1, 0, 0, 0)

    def test_two_jobs_until_completion(self):
        j = self.small_job_parameters.sample(1)
        j.execution_time = 5
        self.scheduler.submit(j)
        self.assertQueuesSane(0, 0, 0, 0, 1)
        self.scheduler.step()
        self.assertEqual(self.scheduler.free_resources[0], self.scheduler.number_of_processors - j.processors_allocated)
        self.assertQueuesSane(1, 0, 1, 0, 0)
        j = self.small_job_parameters.sample(1)
        j.execution_time = 5
        self.scheduler.submit(j)
        self.assertQueuesSane(1, 0, 1, 0, 1)
        self.scheduler.step(3)
        self.assertQueuesSane(4, 0, 2, 0, 0)
        self.scheduler.step()

    def test_should_fail_to_decrease_time(self):
        with self.assertRaises(AssertionError):
            self.scheduler.step(-1)

    def test_should_find_time_for_job_when_cluster_empty(self):
        j = self.small_job_parameters.sample(1)
        self.assertEqual(self.scheduler.current_time, self.scheduler.find_first_time_for(j)[0])

    def test_should_find_time_in_future_when_cluster_busy(self):
        for i in range(100):
            j = self.workload.sample(i)
            if j:
                self.scheduler.submit(j)
        self.scheduler.step()
        j = self.small_job_parameters.sample(1)
        self.assertNotEqual(self.scheduler.current_time, self.scheduler.find_first_time_for(j))

    def test_submitting_seven_jobs(self):
        j1 = self.make_job(0, 2, 2)
        j2 = self.make_job(1, 2, 1)
        j3 = self.make_job(1, 3, 1)
        j4 = self.make_job(1, 1, 1)
        j5 = self.make_job(1, 4, 2)
        j6 = self.make_job(1, 4, 1)
        j7 = self.make_job(1, 2, 2)

        s = fifo_scheduler.FifoScheduler(3, 999999)

        s.submit(j1)
        s.schedule()
        s.step()

        s.submit(j2)
        s.submit(j3)
        s.submit(j4)
        s.submit(j5)
        s.submit(j6)
        s.submit(j7)

        s.schedule()

        self.assertEqual(0, j1.start_time)
        self.assertEqual(1, j2.start_time)
        self.assertEqual(2, j3.start_time)
        self.assertEqual(2, j4.start_time)
        self.assertEqual(3, j5.start_time)
        self.assertEqual(5, j6.start_time)
        self.assertEqual(7, j7.start_time)

        s.step(8)
        self.assertEqual(7, s.makespan)

        s.step(1)
        self.assertEqual(9, s.makespan)

    def test_submitting_invalid_job_fails(self):
        j = self.make_job(0, 2, 256)
        with self.assertRaises(RuntimeError):
            self.scheduler.submit(j)


class TestBinomialWorkloadGenerator(unittest.TestCase):
    def setUp(self):
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)

    def test_that_sampling_stops(self):
        w = workload.BinomialWorkloadGenerator(
            0.7, 0.8, self.small_job_parameters, self.large_job_parameters, length=1
        )
        self.assertEqual(1, len([j for j in w]))

    def test_that_sampling_generates_nones(self):
        w = workload.BinomialWorkloadGenerator(
            0.7, 0.8, self.small_job_parameters, self.large_job_parameters, length=100
        )
        jobs = [j for j in w]
        self.assertTrue(None in jobs)


class TestResourcePool(unittest.TestCase):
    def setUp(self) -> None:
        self.max_size = 32
        self.resource_pool = resource_pool.ResourcePool(resource_pool.ResourceType.CPU, self.max_size)

    def test_zero_used_resources(self):
        self.assertEqual(0, self.resource_pool.used_resources)
        self.assertEqual(self.max_size, self.resource_pool.free_resources)

    def test_job_of_size_zero_fails_to_fit(self):
        with self.assertRaises(AssertionError):
            self.resource_pool.fits(0)

    def test_jobs_of_size_up_to_max_fit(self):
        for size in range(1, self.max_size + 1):
            self.assertTrue(self.resource_pool.fits(size))

    def test_jobs_bigger_than_resource_pool_size_do_not_fit(self):
        self.assertFalse(self.resource_pool.fits(self.max_size + 1))

    def test_can_allocate_size_of_resource(self):
        t = self.resource_pool.find(self.max_size)
        self.assertEqual(1, len(t))
        self.assertEqual(0, t.begin())
        self.assertEqual(self.max_size, t.end())

    def test_cant_find_size_bigger_than_resources(self):
        self.assertEqual(0, len(self.resource_pool.find(self.max_size + 1)))

    def test_should_find_resources_smaller_than_pool_size(self):
        t = self.resource_pool.find(self.max_size - 1)
        self.assertEqual(1, len(t))
        self.assertEqual(0, t.begin())
        self.assertEqual(self.max_size - 1, t.end())

    def test_should_allocate_resources_smaller_than_size(self):
        t = self.resource_pool.find(self.max_size - 1)
        self.resource_pool.allocate(t)
        self.assertEqual(self.max_size - 1, self.resource_pool.used_resources)
        self.assertEqual(1, self.resource_pool.free_resources)

    def test_should_allocate_series_of_one_resource(self):
        for i in range(self.max_size):
            t = self.resource_pool.find(1)
            self.resource_pool.allocate(t)
        self.assertEqual(self.max_size, self.resource_pool.used_resources)
        self.assertEqual(0, self.resource_pool.free_resources)

    def test_should_fail_to_allocate_more_resources(self):
        with self.assertRaises(AssertionError):
            self.resource_pool.allocate([resource_pool.Interval(0, 33)])

    def test_should_deallocate_after_allocation(self):
        t = self.resource_pool.find(1)
        self.resource_pool.allocate(t)
        self.resource_pool.deallocate(t)
        self.assertEqual(0, self.resource_pool.used_resources)
        self.assertEqual(self.max_size, self.resource_pool.free_resources)

    def test_should_have_correct_number_of_intervals(self):
        intervals = []
        for i in range(self.max_size):
            t = self.resource_pool.find(1)
            intervals.append(t)
            self.resource_pool.allocate(t)
        self.assertEqual(len(intervals), len(self.resource_pool.intervals))

    def test_should_revert_state_to_original_after_cleaning_intervals(self):
        intervals = []
        for i in range(0, self.max_size, 2):
            t = self.resource_pool.find(1)
            intervals.append(t)
            self.resource_pool.allocate(t)
        for i in intervals:
            self.resource_pool.deallocate(i)
        self.assertEqual(0, self.resource_pool.used_resources)
        self.assertEqual(self.max_size, self.resource_pool.free_resources)

    def test_should_fail_to_remove_missing_resource(self):
        intervals = []
        for i in range(0, self.max_size, 2):
            t = self.resource_pool.find(1)
            intervals.append(t)
            self.resource_pool.allocate(t)
        for i in intervals:
            self.resource_pool.deallocate(i)
        with self.assertRaises(AssertionError):
            self.resource_pool.deallocate(intervals[0])

    def test_should_have_two_sets_after_allocation_deallocation_allocation(self):
        r1 = self.resource_pool.find(self.max_size // 4)
        self.resource_pool.allocate(r1)
        self.assertEqual(1, len(self.resource_pool.intervals))
        r2 = self.resource_pool.find(self.max_size // 4)
        self.resource_pool.deallocate(r1)
        self.assertEqual(0, len(self.resource_pool.intervals))
        self.resource_pool.allocate(r2)
        self.assertEqual(1, len(self.resource_pool.intervals))
        r3 = self.resource_pool.find(self.max_size // 2)
        self.resource_pool.allocate(r3)
        self.assertEqual(3, len(self.resource_pool.intervals))
        self.resource_pool.deallocate(r3)
        self.assertEqual(1, len(self.resource_pool.intervals))


class TestEvent(unittest.TestCase):
    req: event.EventQueue[event.ResourceEvent]

    def setUp(self) -> None:
        self.req = event.EventQueue()

    @staticmethod
    def build_event(type, interval, time=0):
        t = resource_pool.IntervalTree([resource_pool.Interval(interval[0], interval[1])])
        re = event.ResourceEvent(
            time, type, resource_pool.ResourceType.CPU, t
        )
        return re

    def test_add_event(self):
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2))
        self.req.add(re)
        self.assertEqual(re, self.req.next)

    def test_time_passing(self):
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2))
        self.req.add(re)
        present = list(self.req.step())
        self.assertEqual(present[0], re)
        self.assertEqual(None, self.req.next)
        self.assertEqual(0, len(self.req.future))
        self.assertEqual(1, len(self.req.past))
        self.assertEqual(self.req.last, re)

    def test_double_step_with_zero_yields_no_present_updates(self):
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2))
        self.req.add(re)
        present = list(self.req.step())
        self.assertEqual(1, len(present))
        present = list(self.req.step(0))
        self.assertEqual(0, len(present))

    def test_future_does_not_leak_in(self):
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2), 1)
        self.req.add(re)
        present = list(self.req.step(0))
        self.assertEqual(0, len(present))

    def test_leap_into_the_future(self):
        for i in range(100):
            re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2), i)
            self.req.add(re)
        present = list(self.req.step(101))
        self.assertEqual(100, len(present))
        self.assertEqual(0, len(self.req.future))
        self.assertEqual(100, len(self.req.past))

    def test_step_into_the_past_should_fail(self):
        with self.assertRaises(AssertionError):
            self.req.step(-1)

    def test_add_event_in_the_past_should_work(self):
        self.assertEqual(0, len(list(self.req.step(100))))
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2), 1)
        self.req.add(re)
        self.assertEqual(1, len(self.req.past))
        self.assertEqual(0, len(self.req.future))

    def test_event_in_the_past_should_not_leak_into_present(self):
        self.assertEqual(0, len(list(self.req.step(100))))
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2), 1)
        self.req.add(re)
        present = list(self.req.step())
        self.assertEqual(0, len(present))


class TestHeap(unittest.TestCase):
    def setUp(self) -> None:
        self.heap: heap.Heap[int] = heap.Heap()

    def test_should_change_priority(self):
        item = 42
        self.heap.add(item, 0)
        self.heap.add(item, 10)
        self.assertEqual(1, len(self.heap))

    def test_removing_from_empty_heap_should_fail(self):
        with self.assertRaises(KeyError):
            self.heap.pop()

    def test_should_iterate_over_elements(self):
        for i in range(1000):
            self.heap.add(i, i)
        self.assertEqual(list(sorted(self.heap)), list(range(1000)))

    def test_should_contain_element(self):
        self.heap.add(0, 0)
        self.assertTrue(0 in self.heap)

    def test_should_not_contain_missing_element(self):
        self.heap.add(0, 0)
        self.assertFalse(1 in self.heap)


class TestJob(unittest.TestCase):
    def setUp(self):
        self.counter = 0

    def make_job(self, submission, duration, processors):
        self.counter += 1
        return job.Job(self.counter, submission, duration, processors, 1, 1, processors, duration, 1,
                       job.JobStatus.SCHEDULED, 1, 1, 1, 1, 1, -1, -1, 0)

    def test_slowdown_of_unfinished_job_should_fail(self):
        j = self.make_job(0, 1, 2)
        j.finish_time = None
        self.assertEqual(-1, j.slowdown())

    def test_slowdown_of_atomic_idealized_job(self):
        j = self.make_job(0, 1, 2)
        j.finish_time = 1
        self.assertEqual(1, j.slowdown())
