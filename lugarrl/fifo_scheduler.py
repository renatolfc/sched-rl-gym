#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .job import JobStatus
from .scheduler import Scheduler


class FifoScheduler(Scheduler):
    def schedule(self) -> None:
        for job in self.queue_admission:
            time, resources = self.find_first_time_for(job)
            if not resources.memory or not resources.processors:
                raise AssertionError("Something is terribly wrong")
            job.status = JobStatus.WAITING
            job.resources_used.memory = resources.memory
            job.resources_used.processors = resources.processors
            job.start_time = time
            self.add_job_events(job, time)
            self.queue_waiting.append(job)
        self.queue_admission.clear()
