#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Iterable, Optional

from .job import Job, JobStatus
from .scheduler import Scheduler
from .resource_pool import ResourcePool


class FifoScheduler(Scheduler):
    def schedule(self):
        for job in self.queue_admission:
            time, resources = self.find_first_time_for(job)
            processors = resources.find(job.requested_processors)
            self.add_job_events(job, processors, time)
            job.status = JobStatus.WAITING
            job.processor_list = processors
            self.queue_waiting.append(job)
        self.queue_admission.clear()

    def find_first_time_for(self, job: Job) -> Tuple[int, ResourcePool]:
        resources, pool = self.fits(self.current_time, job, self.processor_pool.clone(), self.job_events)
        if resources:
            return self.current_time, pool

        for event in self.job_events:
            near_future = [e for e in self.job_events if e.time <= event.time]
            pool = self.play_events(
                near_future,
                self.processor_pool.clone()
            )
            resources = self.fits(event.time, job, pool, self.job_events)
            if resources:
                return event.time, pool
        raise AssertionError('Failed to find time for job, even in the far future.')
