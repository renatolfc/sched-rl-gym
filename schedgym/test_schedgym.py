#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gzip
import random
import tempfile
import unittest
import itertools
import urllib.request

from pathlib import Path
from typing import Union, cast

import gym
import numpy as np

from . import simulator, job, workload, pool, event, heap, scheduler
from . import cluster as clstr
from .envs import workload as env_workload
from .workload import swf_parser
from .envs import base, deeprm_env, compact_env


class MockLugarRL(object):
    def __getattr__(self, _):
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
            0.7,
            0.8,
            self.small_job_parameters,
            self.large_job_parameters,
            length=1,
        )
        self.scheduler = scheduler.FifoScheduler(16, 2048)

    def test_time_based_simulator(self):
        sim = simulator.Simulator.make(
            simulator.SimulationType.TIME_BASED, self.workload, self.scheduler
        )
        self.assertEqual(0, sim.current_time)
        for i in range(100):
            sim.step()
        self.assertEqual(100, sim.current_time)

    def test_all_submission_times_are_different(self):
        sim = simulator.Simulator.make(
            simulator.SimulationType.TIME_BASED, self.workload, self.scheduler
        )
        for i in range(100):
            sim.step()
        prev = -1
        for j in sim.scheduler.all_jobs:
            self.assertNotEqual(j.submission_time, prev)
            prev = j.submission_time

    def test_not_all_jobs_executed(self):
        sim = simulator.Simulator.make(
            simulator.SimulationType.TIME_BASED, self.workload, self.scheduler
        )
        for i in range(100):
            sim.step()
        self.assertNotEqual(
            len(self.scheduler.queue_completed), len(self.scheduler.all_jobs)
        )

    def test_all_jobs_executed(self):
        sim = simulator.Simulator.make(
            simulator.SimulationType.TIME_BASED, self.workload, self.scheduler
        )
        for i in range(100):
            sim.step()
        self.assertNotEqual(
            len(self.scheduler.queue_completed), len(self.scheduler.all_jobs)
        )
        for i in range(100):
            sim.step(False)
        self.assertEqual(
            len(self.scheduler.queue_completed), len(self.scheduler.all_jobs)
        )

    def test_invalid_simulator(self):
        with self.assertRaises(RuntimeError):
            simulator.Simulator.make(
                simulator.SimulationType.EVENT_BASED,
                self.workload,
                self.scheduler,
            )
        with self.assertRaises(RuntimeError):
            simulator.Simulator.make(
                42,  # type: ignore
                self.workload,
                self.scheduler,
            )


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
        p = pool.IntervalTree([pool.Interval(interval[0], interval[1])])
        j.resources.processors = p
        return event.JobEvent(time, type, j)

    def new_job(self, processors, time):
        j = self.jp.sample()
        j.requested_processors = processors
        j.requested_time = time
        return j

    def build_event_pair(self, time, interval, j):
        return (
            self.build_event(event.EventType.JOB_START, j, interval, time),
            self.build_event(
                event.EventType.JOB_FINISH,
                j,
                interval,
                time + j.requested_time,
            ),
        )

    def setUp(self):
        self.scheduler = MockScheduler(10, 10000)
        self.jp = job.JobParameters(1, 2, 1, 2, 1, 2)
        self.events = event.EventQueue()

    def test_fits_empty_pool_without_events(self):
        j = self.jp.sample()
        self.assertTrue(
            self.scheduler.fits(
                0, j, self.scheduler.cluster.clone(), self.events
            ).processors
        )
        j.requested_processors = 10
        self.assertTrue(
            self.scheduler.fits(
                0, j, self.scheduler.cluster.clone(), self.events
            ).processors
        )

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

        self.assertTrue(self.scheduler.can_schedule_now(j))
        self.assertTrue(
            self.scheduler.fits(
                0, j, self.scheduler.cluster.clone(), self.events
            )
        )

    def test_fits_partially_filled_pool_with_no_events(self):
        self.scheduler.cluster.processors.allocate(
            pool.IntervalTree([pool.Interval(0, 6)])
        )
        j = self.jp.sample()
        self.assertTrue(self.scheduler.can_schedule_now(j))
        self.assertTrue(
            self.scheduler.fits(
                0, j, self.scheduler.cluster.clone(), self.events
            )
        )

    def test_doesnt_fit_fully_filled_pool_with_no_events(self):
        self.scheduler.cluster.processors.allocate(
            pool.IntervalTree([pool.Interval(0, 10)])
        )
        j = self.jp.sample()
        self.assertFalse(self.scheduler.can_schedule_now(j))
        self.assertFalse(
            self.scheduler.fits(
                0, j, self.scheduler.cluster.clone(), self.events
            )
        )

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

        self.assertTrue(
            self.scheduler.fits(
                20, j, self.scheduler.cluster.clone(), self.events
            )
        )

    def play_events(self, time):
        for e in (e for e in self.events if e.time <= time):
            if e.type == event.EventType.JOB_START:
                self.scheduler.cluster.processors.allocate(
                    e.job.resources.processors
                )
            else:
                self.scheduler.cluster.processors.free(
                    e.job.resources.processors
                )

    def test_eventually_fits_partially_filled_pool(self):
        for i in range(5):
            alloc, free = self.build_event_pair(
                i, (i * 2, (i + 1) * 2), self.new_job(2, 6)
            )
            self.events.add(alloc)
            self.events.add(free)
        j = self.new_job(2, 3)

        self.play_events(5)
        self.assertFalse(
            self.scheduler.fits(
                5, j, self.scheduler.cluster.clone(), self.events
            )
        )

        self.scheduler = MockScheduler(10, 10000)
        self.play_events(6)
        self.assertTrue(
            self.scheduler.fits(
                6, j, self.scheduler.cluster.clone(), self.events
            )
        )

    def test_should_fail_to_add_malformed_job(self):
        j = self.new_job(1, 0)
        with self.assertRaises(AssertionError):
            self.scheduler._add_job_events(j, 0)

    def test_should_fail_to_play_unsupported_type(self):
        j = self.jp.sample()
        j.requested_processors = 10
        j.requested_time = 5

        alloc, free = self.build_event_pair(5, (0, 10), j)
        alloc.type = event.EventType.RESOURCE_ALLOCATE

        with self.assertRaises(RuntimeError):
            self.scheduler.play_events([alloc, free], self.scheduler.cluster)

    def test_should_fail_to_find_resources_on_empty_cluster_with_large_job(
        self,
    ):
        self.scheduler = MockScheduler(16, 10000)

        j = self.new_job(17, 0)
        self.assertFalse(self.scheduler.cluster.find(j))

        alloc, free = self.build_event_pair(0, (0, 17), j)
        self.assertFalse(
            self.scheduler.fits(0, j, self.scheduler.cluster, [alloc, free])
        )

    def test_empty_cluster_should_return_empty_state(self):
        state, jobs, backlog = self.scheduler.state(10, 10)
        self.assertEqual(-1, jobs[0].requested_processors)
        self.assertEqual(-1, jobs[1].requested_processors)
        self.assertEqual(0, state[0][0][1])
        self.assertEqual(0, state[1][0][1])
        self.assertEqual(0, backlog)


class TestFifoBasedSchedulers(unittest.TestCase):
    def setUp(self):
        self.scheduler = scheduler.FifoScheduler(16, 2048)
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        self.workload = workload.BinomialWorkloadGenerator(
            0.7,
            0.8,
            self.small_job_parameters,
            self.large_job_parameters,
            length=1,
        )
        self.counter = 0

    def make_job(self, submission, duration, processors):
        self.counter += 1
        return job.Job(
            self.counter,
            submission,
            duration,
            processors,
            1,
            1,
            processors,
            duration,
            1,
            job.JobStatus.SCHEDULED,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            0,
        )

    def assertQueuesSane(self, time, completed, running, waiting, admission):
        self.assertEqual(self.scheduler.current_time, time)
        self.assertEqual(len(self.scheduler.queue_completed), completed)
        self.assertEqual(len(self.scheduler.queue_running), running)
        self.assertEqual(len(self.scheduler.queue_waiting), waiting)
        self.assertEqual(len(self.scheduler.queue_admission), admission)

    def test_all_jobs_completed(self):
        for i in range(100):
            self.scheduler.step()
            for j in self.workload.step():
                if j:
                    self.scheduler.submit(j)
        for i in range(
            max([j.submission_time for j in self.scheduler.all_jobs]) + 1000
        ):
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
        self.assertEqual(
            self.scheduler.free_resources[0],
            self.scheduler.number_of_processors - j.processors_allocated,
        )
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
        self.assertEqual(
            self.scheduler.current_time,
            self.scheduler.find_first_time_for(j)[0],
        )

    def test_should_find_time_in_future_when_cluster_busy(self):
        for i in range(100):
            for j in self.workload.step():
                if j:
                    self.scheduler.submit(j)
        self.scheduler.step()
        j = self.small_job_parameters.sample(1)
        self.assertNotEqual(
            self.scheduler.current_time, self.scheduler.find_first_time_for(j)
        )

    def test_easy_submitting_six_jobs(self):
        j1 = self.make_job(0, 2, 2)
        j2 = self.make_job(1, 2, 1)
        j3 = self.make_job(1, 2, 3)
        j4 = self.make_job(1, 1, 2)
        j5 = self.make_job(1, 1, 2)
        j6 = self.make_job(3, 1, 2)

        s = scheduler.EasyScheduler(3, 999999)

        s.submit(j1)
        s.step()

        s.submit(j2)
        s.submit(j3)
        s.step()
        s.submit(j4)
        s.submit(j5)
        s.step(2)
        s.submit(j6)
        s.step(2)

        self.assertEqual(0, j1.start_time)
        self.assertEqual(1, j2.start_time)
        self.assertEqual(3, j3.start_time)
        self.assertEqual(2, j4.start_time)
        self.assertEqual(5, j5.start_time)
        self.assertEqual(6, j6.start_time)

        for j in [j1, j2, j3, j4]:
            for i in j.resources.processors:
                self.assertEqual(j.id, i.data)
            for i in j.resources.memory:
                self.assertEqual(j.id, i.data)

        s.step(5)
        self.assertEqual(7, s.makespan)

    def test_fifo_submitting_four_jobs(self):
        j1 = self.make_job(0, 2, 2)
        j2 = self.make_job(1, 2, 1)
        j3 = self.make_job(1, 2, 3)
        j4 = self.make_job(1, 1, 2)

        s = scheduler.FifoScheduler(3, 999999)

        s.submit(j1)
        s.schedule()
        s.step()

        s.submit(j2)
        s.submit(j3)
        s.schedule()
        s.step()
        s.submit(j4)
        s.step(4)

        self.assertEqual(0, j1.start_time)
        self.assertEqual(1, j2.start_time)
        self.assertEqual(3, j3.start_time)
        self.assertEqual(5, j4.start_time)

        for j in [j1, j2, j3, j4]:
            for i in j.resources.processors:
                self.assertEqual(j.id, i.data)
            for i in j.resources.memory:
                self.assertEqual(j.id, i.data)

        s.step(2)
        self.assertEqual(6, s.makespan)

    def test_backfilling_submitting_seven_jobs(self):
        j1 = self.make_job(0, 2, 2)
        j2 = self.make_job(1, 2, 1)
        j3 = self.make_job(1, 3, 1)
        j4 = self.make_job(1, 1, 1)
        j5 = self.make_job(1, 4, 2)
        j6 = self.make_job(1, 4, 1)
        j7 = self.make_job(1, 2, 2)

        s = scheduler.BackfillingScheduler(3, 999999)

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

        for j in [j1, j2, j3, j4, j5, j6, j7]:
            for i in j.resources.processors:
                self.assertEqual(j.id, i.data)
            for i in j.resources.memory:
                self.assertEqual(j.id, i.data)

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

    def test_that_sampling_generates_empty_sequences(self):
        w = workload.BinomialWorkloadGenerator(
            0.0,
            0.8,
            self.small_job_parameters,
            self.large_job_parameters,
            length=100,
        )
        self.assertEqual(0, len(w.step()))


class TestResourcePool(unittest.TestCase):
    def setUp(self) -> None:
        self.max_size = 32
        self.resource_pool = pool.ResourcePool(
            pool.ResourceType.CPU, self.max_size
        )

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
            self.resource_pool.allocate([pool.Interval(0, 33)])

    def test_should_deallocate_after_allocation(self):
        t = self.resource_pool.find(1)
        self.resource_pool.allocate(t)
        self.resource_pool.free(t)
        self.assertEqual(0, self.resource_pool.used_resources)
        self.assertEqual(self.max_size, self.resource_pool.free_resources)

    def test_should_have_correct_number_of_intervals(self):
        intervals = []
        for i in range(self.max_size):
            t = self.resource_pool.find(1)
            intervals.append(t)
            self.resource_pool.allocate(t)
        self.assertEqual(len(intervals), len(self.resource_pool.intervals))

    def add_and_remove_intervals(self):
        intervals = []
        for i in range(0, self.max_size, 2):
            t = self.resource_pool.find(1)
            intervals.append(t)
            self.resource_pool.allocate(t)
        for i in intervals:
            self.resource_pool.free(i)
        return intervals

    def test_should_revert_state_to_original_after_cleaning_intervals(self):
        self.add_and_remove_intervals()
        self.assertEqual(0, self.resource_pool.used_resources)
        self.assertEqual(self.max_size, self.resource_pool.free_resources)

    def test_should_fail_to_remove_missing_resource(self):
        intervals = self.add_and_remove_intervals()
        with self.assertRaises(AssertionError):
            self.resource_pool.free(intervals[0])

    def test_should_have_two_sets_after_allocation_deallocation_allocation(
        self,
    ):
        r1 = self.resource_pool.find(self.max_size // 4)
        self.resource_pool.allocate(r1)
        self.assertEqual(1, len(self.resource_pool.intervals))
        r2 = self.resource_pool.find(self.max_size // 4)
        self.resource_pool.free(r1)
        self.assertEqual(0, len(self.resource_pool.intervals))
        self.resource_pool.allocate(r2)
        self.assertEqual(1, len(self.resource_pool.intervals))
        r3 = self.resource_pool.find(self.max_size // 2)
        self.resource_pool.allocate(r3)
        self.assertEqual(3, len(self.resource_pool.intervals))
        self.resource_pool.free(r3)
        self.assertEqual(1, len(self.resource_pool.intervals))


class TestEvent(unittest.TestCase):
    req: event.EventQueue[event.ResourceEvent]

    def setUp(self) -> None:
        self.req = event.EventQueue()

    @staticmethod
    def build_event(type, interval, time=0):
        t = pool.IntervalTree([pool.Interval(interval[0], interval[1])])
        re = event.ResourceEvent(time, type, pool.ResourceType.CPU, t)
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
        with self.assertWarns(UserWarning):
            self.req.add(re)
        self.assertEqual(1, len(self.req.past))
        self.assertEqual(0, len(self.req.future))

    def test_event_in_the_past_should_not_leak_into_present(self):
        self.assertEqual(0, len(list(self.req.step(100))))
        re = self.build_event(event.EventType.RESOURCE_ALLOCATE, (0, 2), 1)
        with self.assertWarns(UserWarning):
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
        return job.Job(
            self.counter,
            submission,
            duration,
            processors,
            1,
            1,
            processors,
            duration,
            1,
            job.JobStatus.SCHEDULED,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            0,
        )

    def test_slowdown_of_unfinished_job_should_fail(self):
        j = self.make_job(0, 1, 2)
        j.finish_time = -1
        with self.assertWarns(UserWarning):
            self.assertEqual(-1, j.slowdown)

    def test_slowdown_of_atomic_idealized_job(self):
        j = self.make_job(0, 1, 2)
        j.finish_time = 1
        self.assertEqual(1, j.slowdown)


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.counter = 0
        self.processors = 16
        self.memory = 1024 * self.processors

    def make_job(self, submission, duration, processors, memory):
        self.counter += 1
        return job.Job(
            self.counter,
            submission,
            duration,
            processors,
            1,
            memory,
            processors,
            duration,
            memory,
            job.JobStatus.SCHEDULED,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            0,
        )

    def test_basic_fit(self):
        cluster = clstr.Cluster(self.processors, self.memory)
        j = self.make_job(0, 10, 1, 1024)
        self.assertTrue(cluster.fits(j))
        j = self.make_job(0, 10, self.processors + 1, 1024)
        self.assertFalse(cluster.fits(j))
        j = self.make_job(0, 10, self.processors, self.memory)
        self.assertTrue(cluster.fits(j))
        j = self.make_job(0, 10, self.processors, self.memory + 1)
        self.assertFalse(cluster.fits(j))

    def test_fit_ignoring_memory(self):
        cluster = clstr.Cluster(
            self.processors, self.memory, ignore_memory=True
        )
        self.assertTrue(cluster.fits(self.make_job(0, 10, 1, 1024)))
        self.assertTrue(
            cluster.fits(self.make_job(0, 10, self.processors, 1024))
        )
        self.assertTrue(
            cluster.fits(self.make_job(0, 10, self.processors, self.memory))
        )
        self.assertTrue(
            cluster.fits(
                self.make_job(0, 10, self.processors, self.memory + 1)
            )
        )
        self.assertFalse(
            cluster.fits(
                self.make_job(0, 10, self.processors + 1, self.memory + 1)
            )
        )

    def test_free_resources(self):
        cluster = clstr.Cluster(self.processors, self.memory)
        self.assertEqual(
            cluster.free_resources, (self.processors, self.memory)
        )

    def test_allocation(self):
        cluster = clstr.Cluster(self.processors, self.memory)
        j = self.make_job(0, 10, 1, 1024)
        j.resources.processors = pool.IntervalTree([pool.Interval(0, 1)])
        j.resources.memory = pool.IntervalTree([pool.Interval(0, 1024)])
        cluster.allocate(j)
        self.assertEqual(cluster.free_resources[0], self.processors - 1)
        self.assertEqual(cluster.free_resources[1], self.memory - 1024)
        cluster.free(j)
        self.assertEqual(cluster.free_resources[0], self.processors)
        self.assertEqual(cluster.free_resources[1], self.memory)
        j = self.make_job(0, 10, self.processors + 1, 1024)
        with self.assertRaises(AssertionError):
            cluster.allocate(j)
        j = self.make_job(0, 10, self.processors, self.memory)
        j.resources.processors = pool.IntervalTree(
            [pool.Interval(0, self.processors)]
        )
        j.resources.memory = pool.IntervalTree([pool.Interval(0, self.memory)])
        j.ignore_memory = False
        cluster.allocate(j)
        self.assertEqual(cluster.free_resources, (0, 0))
        cluster.free(j)
        j.requested_memory = self.memory + 1
        with self.assertRaises(AssertionError):
            cluster.allocate(j)


class TestSchedulers(unittest.TestCase):
    def setUp(self):
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        self.workload = workload.BinomialWorkloadGenerator(
            1.0,
            0.8,
            self.small_job_parameters,
            self.large_job_parameters,
            length=1,
        )

    def submit_jobs(self, s: scheduler.Scheduler, n: int):
        for _ in range(n):
            j = self.workload.step()
            if j:
                s.submit(j)

    def test_packer_scheduler(self):
        s = scheduler.PackerScheduler(16, 2048)
        self.submit_jobs(s, 10)
        top = max(
            s.queue_admission,
            key=lambda j: s.free_resources[0] * j.requested_processors
            + s.free_resources[1] * j.requested_memory,
        )
        s.schedule()
        self.assertEqual(top, s.queue_waiting[0])

    def test_null_scheduler_invariant(self):
        s = scheduler.NullScheduler(16, 2048)
        self.submit_jobs(s, 10)
        s.current_slot = -1
        with self.assertRaises(AssertionError):
            s.step()

    def test_null_scheduler(self):
        s = scheduler.NullScheduler(16, 2048)
        self.assertEqual(s.action_space, 1)
        self.submit_jobs(s, 10)
        self.assertEqual(s.sjf_action(1), 0)

        # Tests whether it finds the job with smallest time
        jobs = [(j.requested_time, i) for i, j in enumerate(s.all_jobs)]
        jobs.sort()
        self.assertEqual(s.sjf_action(-1), jobs[0][-1])

        s.step(0)
        self.assertEqual(s.all_jobs[0].status, job.JobStatus.WAITING)
        s.forward_time()
        self.assertEqual(s.all_jobs[0].status, job.JobStatus.RUNNING)

    def test_sjf_scheduler(self):
        s = scheduler.SjfScheduler(16, 2048)
        self.submit_jobs(s, 10)
        top = min(s.queue_admission, key=lambda j: j.requested_time)
        s.schedule()
        self.assertEqual(top, s.queue_waiting[0])

    def test_tetris_scheduler_behaving_like_packer(self):
        s = scheduler.TetrisScheduler(64, 2048, 1.0)
        self.submit_jobs(s, 10)
        top = max(
            s.queue_admission,
            key=lambda j: s.free_resources[0] * j.requested_processors
            + s.free_resources[1]
            + j.requested_memory,
        )
        s.schedule()
        self.assertEqual(
            s.get_priority(top), s.get_priority(s.queue_waiting[0])
        )

    def test_tetris_scheduler_behaving_like_sjf(self):
        s = scheduler.TetrisScheduler(64, 2048, 0.0)
        self.submit_jobs(s, 10)
        top = min(s.queue_admission, key=lambda j: j.requested_time)
        s.schedule()
        self.assertEqual(
            s.get_priority(top), s.get_priority(s.queue_waiting[0])
        )

    def test_tetris_scheduler(self):
        s = scheduler.TetrisScheduler(64, 2048, 0.5)
        self.submit_jobs(s, 10)
        top = max(
            s.queue_admission,
            key=lambda j: 0.5 / j.requested_time
            + 0.5
            * (
                s.free_resources[0] * j.requested_processors
                + s.free_resources[1]
                + j.requested_memory
            ),
        )
        s.schedule()
        self.assertEqual(
            s.get_priority(top), s.get_priority(s.queue_waiting[0])
        )

    def test_random_scheduler(self):
        s = scheduler.RandomScheduler(16, 2048)
        self.submit_jobs(s, 10)
        s.step(100)
        self.assertEqual(0, len(s.queue_admission))
        self.assertEqual(0, len(s.queue_waiting))

    def test_scheduler_state(self):
        total_memory = 2048
        total_processors = 16

        timesteps = 100
        job_slots = 4

        s = scheduler.SjfScheduler(total_processors, total_memory)
        self.submit_jobs(s, 20)
        state, jobs, backlog = s.state(timesteps, job_slots)

        self.assertEqual(timesteps, len(state[0]))
        self.assertEqual(total_processors, state[0][0][0])
        self.assertEqual(total_memory, state[1][0][0])

        self.assertEqual(job_slots, len(jobs))

        # state is empty because we haven't stepped the simulator
        self.assertEqual(0, state[0][0][1])
        self.assertEqual(0, state[1][0][1])

        # Everything must be in the backlog + job slots
        self.assertEqual(len(s.all_jobs) - job_slots, backlog)

        for _ in range(5):
            s.step()
            state, jobs, backlog = s.state(timesteps, job_slots)

            self.assertEqual(
                max(len(s.queue_admission) - job_slots, 0), backlog
            )
            for j in s.queue_admission[:job_slots]:
                self.assertEqual(
                    j.requested_time, jobs[j.slot_position].requested_time
                )
                self.assertEqual(
                    j.requested_memory, jobs[j.slot_position].requested_memory
                )
                self.assertEqual(
                    j.requested_processors,
                    jobs[j.slot_position].requested_processors,
                )


class TestSwfGenerator(unittest.TestCase):
    TOTAL_JOBS = 122052
    TEST_DIR = 'test'
    TRACE_FILE = 'LANL-CM5-1994-4.1-cln.swf.gz'

    def setUp(self) -> None:
        self.tracefile = Path(self.TEST_DIR) / self.TRACE_FILE
        try:
            open(self.tracefile).close()
        except FileNotFoundError:
            Path(self.TEST_DIR).mkdir(exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                dir=self.TEST_DIR, mode='wb', delete=False
            )
            data = urllib.request.urlopen(
                'http://www.cs.huji.ac.il/labs/parallel/workload/l_lanl_cm5/'
                f'{self.TRACE_FILE}',
            )
            tmp.write(gzip.decompress(data.read()))
            tmp.close()
            os.rename(tmp.name, self.tracefile)  # atomic in same fs

    def load(self, offset=0, length=None):
        return workload.SwfGenerator(
            self.tracefile,
            1024,
            1024,
            length=length,
            offset=offset,
        )

    def test_parsing(self):
        jobs = list(swf_parser.parse(self.tracefile, 1024, 1024))
        self.assertEqual(self.TOTAL_JOBS, len(jobs))

    def test_workload_length(self):
        wl = self.load()
        self.assertEqual(self.TOTAL_JOBS, len(list(wl)))

    def test_stepping(self):
        wl = self.load()
        initial = wl.trace[0].submission_time + 1
        jobs = wl.step(initial)
        self.assertEqual(1, len(jobs))
        jobs = wl.step()
        self.assertEqual(0, len(jobs))
        jobs = wl.step(wl.trace[2].submission_time - initial + 1)
        self.assertEqual(2, len(jobs))

    def test_internals(self):
        wl = self.load()
        wl.step()
        self.assertEqual(1, wl.current_element)

    def test_get_all(self):
        wl = self.load()
        wl.step(int(1e9))
        self.assertEqual(self.TOTAL_JOBS, wl.current_element)
        with self.assertRaises(StopIteration):
            wl.step()

    def test_get_all_restarts(self):
        wl = self.load()
        wl.step(int(1e9))
        wl.restart = True
        wl.step()
        self.assertEqual(1, wl.current_element)
        wl.step(int(1e9))
        self.assertEqual(self.TOTAL_JOBS, wl.current_element)

    def test_fixed_length(self):
        wl = self.load(length=10)
        self.assertEqual(10, len(list(wl)))

    def test_offset_length(self):
        wl = self.load(offset=1, length=10)
        self.assertEqual(10, len(list(wl)))

    def test_iterating(self):
        wl = self.load(offset=1, length=10)
        for j in wl:
            self.assertIsNotNone(j)
        with self.assertRaises(StopIteration):
            while True:
                next(wl)

    def test_last_event_time(self):
        wl = self.load(offset=1, length=1)
        self.assertEqual(
            wl.last_event_time,
            cast(job.Job, wl.peek()).submission_time
        )
        j = wl.step(wl.last_event_time + 1)[0]
        with self.assertRaises(StopIteration):
            wl.step()
        self.assertEqual(wl.last_event_time, j.submission_time)


class TestEnvWorkload(unittest.TestCase):
    def test_distribution_factory(self):
        config = {
            'type': 'deeprm',
            'new_job_rate': 1.0,
            'small_job_chance': 0.0,
            'max_job_len': 10,
            'max_job_size': 10,
        }
        wl = env_workload.build(config)
        j = cast(job.Job, wl.step()[0])
        self.assertLessEqual(j.requested_time, 10)


class TestBaseEnv(unittest.TestCase):
    def test_rewardjobs_parsing(self):
        with self.assertRaises(ValueError):
            base.RewardJobs.from_str('none')
        for string in self.all_casings('all'):
            self.assertEqual(
                base.RewardJobs.from_str(string), base.RewardJobs.ALL
            )
        for string in self.all_casings('waiting'):
            self.assertEqual(
                base.RewardJobs.from_str(string), base.RewardJobs.WAITING
            )
        for string in self.all_casings('job_slots'):
            self.assertEqual(
                base.RewardJobs.from_str(string), base.RewardJobs.JOB_SLOTS
            )

    @staticmethod
    def all_casings(string):
        return list(
            map(
                ''.join,
                itertools.product(*zip(string.upper(), string.lower())),
            )
        )


class TestCompactEnv(unittest.TestCase):
    @staticmethod
    def sjf_action(env, observation) -> int:
        """Selects the job SJF (Shortest Job First) would select."""

        skip = env.time_horizon * 2 * (1 if env.ignore_memory else 2)
        size = len(job.JobState._fields)
        time = job.JobState._fields.index('requested_time')
        reqs = observation[skip:][time:(env.job_slots * size):size]
        reqs[reqs == 0] = 1.1
        action = np.argmin(reqs)
        return int(action)

    def test_instantiation_with_gym(self):
        gym.make('CompactRM-v0')

    def test_environment_with_sjf_agent_to_completion(self):
        for simulation_type in 'time-based event-based'.split():
            env: compact_env.CompactRmEnv = gym.make(  # type: ignore
                'CompactRM-v0', simulation_type=simulation_type
            )
            observation = env.reset()
            action = 0
            done = False
            while not done:
                observation, _, done, extra = env.step(action)
                action = self.sjf_action(env, observation)
                submission_times = [
                    j.execution_time
                    for j in env.scheduler.queue_admission[: env.job_slots]
                ]
                self.assertEqual(
                    action,
                    np.argmin(submission_times) if submission_times else 0,
                )

    def test_scheduler_identity(self):
        env: deeprm_env.DeepRmEnv = gym.make(  # type: ignore
            'CompactRM-v0',
            **{'use_raw_state': True, 'simulation_type': 'event-based'},
        )
        _ = env.reset()
        self.assertEqual(id(env.scheduler), id(env.simulator.scheduler))

    def test_synthetic_wl_auto_time_limit(self):
        env: compact_env.CompactRmEnv = gym.make(  # type: ignore
            'CompactRM-v0',
            **{
                'use_raw_state': True,
                'time_limit': None,
                'simulation_type': 'event-based',
                'ignore_memory': True,
                'workload': {
                    'type': 'lublin',
                    'length': 10,
                    'nodes': 10,
                }
            },
        )
        obs = env.reset()
        self.assertNotIn(None, obs)
        obs, _, _, _ = env.step(0)
        self.assertNotIn(None, obs)


class TestDeepRmEnv(unittest.TestCase):
    def test_instantiation_with_gym(self):
        gym.make('DeepRM-v0')

    def test_environment_with_trivial_agent(self):
        for simulation_type in 'time-based event-based'.split():
            env: deeprm_env.DeepRmEnv = gym.make(  # type: ignore
                'DeepRM-v0',
                **{
                    'use_raw_state': True,
                    'simulation_type': simulation_type,
                },
            )
            _ = env.reset()
            action = 0
            done = False
            while not done:
                _, _, done, _ = env.step(action)
            self.assertTrue(done)

    def test_environment_with_event_based_simulator(self):
        env: deeprm_env.DeepRmEnv = gym.make(  # type: ignore
            'DeepRM-v0',
            **{'use_raw_state': True, 'simulation_type': 'event-based'},
        )
        _ = env.reset()
        action = 0
        done = False
        while not done:
            _, _, done, _ = env.step(action)
        self.assertTrue(done)

    def test_scheduler_identity(self):
        env: deeprm_env.DeepRmEnv = gym.make(  # type: ignore
            'DeepRM-v0',
            **{'use_raw_state': True, 'simulation_type': 'event-based'},
        )
        _ = env.reset()
        self.assertEqual(id(env.scheduler), id(env.simulator.scheduler))



class TestRewardMappers(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = 0

    def make_job(self, submission, duration, processors):
        self.counter += 1
        return job.Job(
            self.counter,
            submission,
            duration,
            processors,
            1,
            1,
            processors,
            duration,
            1,
            job.JobStatus.SCHEDULED,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            0,
        )

    def test_instantiation_and_reward_computation(self):
        total_jobs = 100
        scheduled_jobs = 5
        execution_time = 10
        reward_methods = ['all', 'waiting', 'job-slots', 'running-job-slots']
        for e in 'DeepRM-v0 CompactRM-v0'.split():
            for m, mapper in enumerate(reward_methods):
                env: Union[
                    compact_env.CompactRmEnv, deeprm_env.DeepRmEnv
                ] = gym.make(
                    e,
                    reward_jobs=mapper,
                    workload=dict(
                        type='deeprm',
                        ignore_memory=False,
                        new_job_rate=0.0,
                        small_job_chance=0.0,
                        max_job_len=15,
                        max_job_size=10,
                    ),
                )
                rewards = [
                    -np.sum([1 / execution_time for _ in range(total_jobs)]),
                    -np.sum(
                        [
                            1 / execution_time
                            for _ in range(total_jobs - scheduled_jobs)
                        ]
                    ),
                    -np.sum(
                        [1 / execution_time for _ in range(env.job_slots)]
                    ),
                    -np.sum(
                        [
                            1 / execution_time
                            for _ in range(env.job_slots + scheduled_jobs)
                        ]
                    ),
                ]
                env.reset()
                jobs = [
                    self.make_job(0, execution_time, 1)
                    for i in range(total_jobs)
                ]
                env.scheduler.submit(jobs)
                for i in range(scheduled_jobs):
                    env.step(0)
                    self.assertEqual(env.simulator.current_time, 0)
                env.step(env.job_slots)  # forward time
                self.assertEqual(env.simulator.current_time, 1)
                self.assertAlmostEqual(env.reward, rewards[m], delta=1e-5)
