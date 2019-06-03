#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from typing import Optional

from lugarrl.job import Job, JobStatus
from . import scheduler


class FifoScheduler(scheduler.Scheduler):
    def first_job(self) -> Optional[Job]:
        first: Optional[Job] = None
        submission_time: int = sys.maxsize
        for job in self.queue_waiting:
            if job.submission_time < submission_time:
                submission_time = job.submission_time
                first = job
        return first

    def schedule(self):
        first: Optional[Job] = self.first_job()
        if first:
            if first.status in (JobStatus.WAITING, JobStatus.SUBMITTED):
                first.status = JobStatus.SCHEDULED
            if self.can_start(first):
                self.start_running(first)
                self.update_queues()
