#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lugarrl.job import JobStatus
from lugarrl.scheduler import Scheduler


class FifoScheduler(Scheduler):
    def schedule(self) -> None:
        for job in self.queue_admission:
            time, resources = self.find_first_time_for(job)
            if not resources:
                raise AssertionError("Something is terribly wrong")
            self.assign_schedule(job, resources, time)
        self.queue_admission.clear()
