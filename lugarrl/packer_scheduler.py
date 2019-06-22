#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .job import Job
from .scheduler import Scheduler


class PackerScheduler(Scheduler):
    """Implements the Packer heuristic.

    In the paper, they give higher priority to the dot product of requested
    resources with the set of available resources. This means that, since
    the number of resources don't change, the jobs that use more resources
    will be preferred. Every time a new job is schedule, the prioritization
    changes, since they always schedule one job at a time.
    """

    def get_priority(self, j: Job) -> int:
        return self.free_resources[0] * j.requested_processors + self.free_resources[1] + j.requested_memory

    def get_highest_priority_job(self) -> Job:
        if not self.queue_admission:
            raise AssertionError("Unable to get highest priority job of empty admission list")
        index: int = 0
        priority: int = 0
        for i, j in enumerate(self.queue_admission):
            if not self.cluster.fits(j):
                continue
            tmp = self.get_priority(j)
            if tmp > priority:
                priority = tmp
                index = i
        return self.queue_admission[index]

    def schedule(self) -> None:
        while any([self.can_schedule_now(j) for j in self.queue_admission]):
            job = self.get_highest_priority_job()
            self.queue_admission.remove(job)
            resources = self.can_schedule_now(job)
            if not resources:
                raise AssertionError("Found job that fits cluster, but couldn't allocate resources.")
            self.assign_schedule(job, resources, self.current_time)

