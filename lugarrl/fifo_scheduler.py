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
        while first and self.can_schedule(first):
            print('Scheduling a job.')
            self.queue_waiting.remove(first)
            first.status = JobStatus.SCHEDULED
            self.start_running(first)
            self.queue_running.append(first)
            first = self.first_job()
