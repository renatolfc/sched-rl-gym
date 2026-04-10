import unittest
from unittest.mock import patch

from schedgym import event, heap, job, scheduler


class TestEventsBetweenState(unittest.TestCase):
    def setUp(self):
        self.scheduler = scheduler.FifoScheduler(16, 2048)

    def _submit_jobs(self):
        jobs = [
            job.Job(1, requested_processors=2, requested_time=3, requested_memory=4),
            job.Job(2, requested_processors=3, requested_time=4, requested_memory=5),
            job.Job(3, requested_processors=1, requested_time=2, requested_memory=1),
        ]
        for item in jobs:
            self.scheduler.submit(item)
        return jobs

    def test_state_shape_is_valid(self):
        self._submit_jobs()
        state, jobs, backlog = self.scheduler.state(5, 4)

        self.assertIsInstance(state, list)
        self.assertEqual(2, len(state))
        self.assertEqual(5, len(state[0]))
        self.assertEqual(4, len(jobs))
        self.assertIsInstance(backlog, int)

    def test_events_between_matches_filter_semantics(self):
        self._submit_jobs()
        current_time = self.scheduler.current_time
        timesteps = 10

        expected = list(
            filter(
                lambda e: e.time < current_time + timesteps,
                self.scheduler.job_events,
            )
        )
        actual = list(
            self.scheduler.job_events.events_between(
                current_time, current_time + timesteps
            )
        )

        self.assertEqual(expected, actual)

    def test_events_at_boundary_are_excluded(self):
        self._submit_jobs()
        boundary = self.scheduler.current_time + 1

        actual = list(
            self.scheduler.job_events.events_between(
                self.scheduler.current_time, boundary
            )
        )

        self.assertTrue(all(e.time < boundary for e in actual))
        self.assertTrue(all(e.time >= self.scheduler.current_time for e in actual))

    def test_zero_events_in_range_is_empty(self):
        self._submit_jobs()
        actual = list(self.scheduler.job_events.events_between(999, 1000))

        self.assertEqual([], actual)

    def test_state_calls_events_between(self):
        self._submit_jobs()
        original = event.EventQueue.events_between
        calls = []

        def spy(self, start, end):
            calls.append((start, end))
            return original(self, start, end)

        with patch.object(event.EventQueue, "events_between", new=spy):
            self.scheduler.state(5, 4)
            self.assertGreaterEqual(len(calls), 1)

    def test_state_does_not_use_heapsort(self):
        self._submit_jobs()
        original = heap.Heap.heapsort
        calls = []

        def spy(self):
            calls.append(self)
            return original(self)

        with patch.object(heap.Heap, "heapsort", new=spy):
            self.scheduler.state(5, 4)
            self.assertEqual(0, len(calls))
