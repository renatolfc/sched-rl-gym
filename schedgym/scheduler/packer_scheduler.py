#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""packer_scheduler - A scheduler based on the Packer heuristic"""

from typing import List

from schedgym.job import Job
from schedgym.scheduler import Scheduler


class PackerScheduler(Scheduler):
    """Implements the Packer heuristic.

    In the paper, they give higher priority to the dot product of requested
    resources with the set of available resources. This means that, since
    the number of resources don't change, the jobs that use more resources
    will be preferred. Every time a new job is scheduled, the prioritization
    changes, since they always schedule one job at a time.
    """

    def get_priority(self, j: Job) -> int:
        """Computes the priority of a given job.

        This computes the priority of a job according to the Packer heuristic,
        which will prefer jobs with higher dot-product between free and
        requested resources.

        Parameters
        ----------
            j : Job
                The job whose priority is to be calculated.
        """
        return (
            self.free_resources[0] * j.requested_processors
            + self.free_resources[1] * j.requested_memory
        )

    def schedule(self) -> None:
        """Schedules jobs according to the Packer heuristic."""
        ignored_jobs: List[Job] = []
        for job in reversed(
            sorted(self.queue_admission, key=lambda j: self.get_priority(j))
        ):
            resources = self.can_schedule_now(job)
            if resources:
                self.assign_schedule(job, resources, self.current_time)
            else:
                ignored_jobs.append(job)
        self.queue_admission = ignored_jobs
