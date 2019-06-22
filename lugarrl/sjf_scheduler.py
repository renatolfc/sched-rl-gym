#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

from .job import Job
from .scheduler import Scheduler


class SjfScheduler(Scheduler):
    def schedule(self) -> None:
        ignored_jobs: List[Job] = []
        # XXX: We always re-sort the queue. If we ever want to learn from demonstration, we'd probably have to
        # do something like:
        # candidates = sorted(enumerate(self.queue_admission), key=lambda e:(e[1].requested_time, e[1].submission_time))
        # and work from there. Hence, the jobs we scheduled would have their indices and we could generate intermediate
        # states as needed.
        for job in sorted(self.queue_admission, key=lambda j: (j.requested_time, j.submission_time)):
            resources = self.can_schedule_now(job)
            if resources:
                self.assign_schedule(job, resources, self.current_time)
            else:
                ignored_jobs.append(job)
        self.queue_admission = ignored_jobs
