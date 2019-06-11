#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Iterable
from collections import defaultdict

from .job import Job, JobStatus
from .scheduler import Scheduler
from .resource_pool import Interval


class FifoScheduler(Scheduler):
    def schedule(self):
        for job in self.queue_admission:
            time, processors = self.find_first_time_for(job)
            job.status = JobStatus.WAITING
            job.processor_list = processors
            job.start_time = time
            self.add_job_events(job, time)
            self.queue_waiting.append(job)
        self.queue_admission.clear()

    def find_first_time_for(self, job: Job) -> Tuple[int, Iterable[Interval]]:
        if (not self.job_events.next) or self.job_events.next.time > self.current_time:
            resources = self.fits(self.current_time, job, self.processor_pool, self.job_events)
            if resources:
                return self.current_time, resources

        near_future = defaultdict(list)
        for e in self.job_events:
            near_future[e.time].append(e)

        resource_pool = self.processor_pool.clone()
        for time in sorted(near_future):
            resource_pool = self.play_events(near_future[time], resource_pool)
            resources = self.fits(time, job, resource_pool, self.job_events)
            if resources:
                return time, resources

        raise AssertionError('Failed to find time for job, even in the far future.')
