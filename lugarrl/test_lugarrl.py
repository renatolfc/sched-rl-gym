#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest

from . import lugarrl, simulator, job, scheduler, workload, fifo_scheduler


class MockLugarRL(object):
    def __getattr__(self, item):
        return lambda: None


class TestLugarRL(unittest.TestCase):
    def setUp(self):
        self.lugar = MockLugarRL()

    def test_single_step(self):
        self.lugar.step()


class TestTimeBaseLugarRL(TestLugarRL):
    def setUp(self):
        self.lugar = lugarrl.LugarRL()

    def test_time_based_simulator_instance(self):
        self.assertTrue(isinstance(self.lugar.simulator, simulator.TimeBasedSimulator))

    def test_step(self):
        self.assertEqual(self.lugar.simulator.current_time, 0)
        self.lugar.step()
        self.assertEqual(self.lugar.simulator.current_time, 1)


class TestJobParameters(unittest.TestCase):
    def setUp(self):
        self.jp = job.JobParameters(1, 2, 1, 2, 1, 2)

    def test_first_job_id_isnt_zero(self):
        j = self.jp.sample()
        self.assertNotEqual(0, j.id)

    def test_any_negative_bounds_should_fail(self):
        with self.assertRaises(AssertionError):
            jp = job.JobParameters(0, 0, 0, 0, 0, 0)

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


class TestScheduler(unittest.TestCase):
    def test_filter_update_queue(self):
        def predicate(j):
            return j.submission_time < 5000

        def update(j):
            j.submission_time = -1

        parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        queue = [parameters.sample(i) for i in range(10000)]
        true_values, false_values = [], []

        for e in queue:
            if predicate(e):
                true_values.append(e)
            else:
                false_values.append(e)

        true_gen, false_gen = scheduler.filter_update_queue(predicate, lambda j: None, queue)
        true_gen = list(true_gen)
        false_gen = list(false_gen)

        self.assertEqual(true_gen, true_values)
        self.assertEqual(false_gen, false_values)

        true_gen, false_gen = scheduler.filter_update_queue(predicate, update, queue)

        for e in true_gen:
            self.assertEqual(-1, e.submission_time)

        self.assertEqual(5000, sum([predicate(j) for j in queue]))


class TestFifoScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = fifo_scheduler.FifoScheduler(8, 64)
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        self.workload = workload.BinomialWorkloadGenerator(
            0.7, 0.8, self.small_job_parameters, self.large_job_parameters, length=1
        )

    def test_can_schedule_small_job(self):
        self.scheduler.step()
        self.assertTrue(
            self.scheduler.can_schedule(self.small_job_parameters.sample(
                self.scheduler.current_time
            ))
        )

    def test_cant_schedule_future_jobs(self):
        self.scheduler.step()
        self.assertFalse(
            self.scheduler.can_schedule(self.small_job_parameters.sample(
                self.scheduler.current_time + 1
            ))
        )


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


