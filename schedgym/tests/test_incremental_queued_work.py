import builtins
import unittest
from unittest.mock import patch

from schedgym.job import Job
from schedgym.scheduler import (
    EasyScheduler,
    FifoScheduler,
    PackerScheduler,
    SjfScheduler,
)


class TestIncrementalQueuedWork(unittest.TestCase):
    def setUp(self):
        self.scheduler = FifoScheduler(16, 2048)

    def make_job(self, job_id: int, requested_processors: int, requested_time: int):
        return Job(
            job_id,
            requested_processors=requested_processors,
            requested_time=requested_time,
            requested_memory=1,
        )

    def test_submit_updates_queued_work_cumulatively(self):
        jobs = [
            self.make_job(1, 2, 3),
            self.make_job(2, 4, 5),
            self.make_job(3, 1, 7),
        ]

        for job in jobs:
            self.scheduler.submit(job)

        self.assertEqual([0, 6, 26], [job.queued_work for job in jobs])

    def test_scheduler_has_incremental_total_attribute_after_init(self):
        self.assertEqual(self.scheduler._queued_work_total, 0)  # type: ignore[attr-defined]

    def test_submit_does_not_call_builtin_sum(self):
        jobs = [self.make_job(4, 1, 2), self.make_job(5, 3, 4)]

        with patch("builtins.sum", wraps=builtins.sum) as mock_sum:
            for job in jobs:
                self.scheduler.submit(job)

        self.assertFalse(mock_sum.called)

    def test_fifo_schedule_updates_incremental_total(self):
        jobs = [self.make_job(6, 1, 2), self.make_job(7, 2, 3)]
        for job in jobs:
            self.scheduler.submit(job)

        self.scheduler.schedule()

        self.assertEqual(0, self.scheduler._queued_work_total)  # type: ignore[attr-defined]

    def test_sjf_schedule_updates_incremental_total(self):
        scheduler = SjfScheduler(16, 2048)
        jobs = [self.make_job(8, 1, 5), self.make_job(9, 2, 1)]
        for job in jobs:
            scheduler.submit(job)

        scheduler.schedule()

        self.assertEqual(0, scheduler._queued_work_total)  # type: ignore[attr-defined]

    def test_easy_schedule_updates_incremental_total(self):
        scheduler = EasyScheduler(16, 2048)
        jobs = [self.make_job(10, 1, 5), self.make_job(11, 2, 1)]
        for job in jobs:
            scheduler.submit(job)

        scheduler.schedule()

        self.assertEqual(0, scheduler._queued_work_total)  # type: ignore[attr-defined]

    def test_packer_schedule_updates_incremental_total(self):
        scheduler = PackerScheduler(16, 2048)
        jobs = [self.make_job(12, 1, 5), self.make_job(13, 2, 1)]
        for job in jobs:
            scheduler.submit(job)

        scheduler.schedule()

        self.assertEqual(0, scheduler._queued_work_total)  # type: ignore[attr-defined]

    def test_backfilling_schedule_resets_incremental_total(self):
        from schedgym.scheduler import BackfillingScheduler

        scheduler = BackfillingScheduler(16, 2048)
        jobs = [self.make_job(14, 1, 5), self.make_job(15, 2, 3)]
        for job in jobs:
            scheduler.submit(job)

        scheduler.schedule()
        self.assertEqual(0, scheduler._queued_work_total)  # type: ignore[attr-defined]

    def test_null_scheduler_schedule_updates_incremental_total(self):
        from schedgym.scheduler import NullScheduler

        scheduler = NullScheduler(16, 2048)
        job = self.make_job(16, 1, 2)
        scheduler.submit(job)
        initial_total = scheduler._queued_work_total  # type: ignore[attr-defined]
        self.assertEqual(2, initial_total)
        scheduler.step(0)
        self.assertEqual(0, scheduler._queued_work_total)  # type: ignore[attr-defined]
