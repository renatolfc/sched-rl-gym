#!/usr/bin/env python
# -*- coding: utf-8 -*-

"fifo_scheduler - Module for a First-In-First-Out scheduler"

from lugarrl.scheduler import Scheduler


class FifoScheduler(Scheduler):
    """Implements a FIFO scheduler, honoring strictly submission order.

    Submission order is honored even if it creates fragmentation.
    """
    def schedule(self) -> None:
        "Schedules a job according to the FIFO strategy"
        for job in self.queue_admission:
            time, resources = self.find_first_time_for(job)
            if not resources:
                raise AssertionError("Something is terribly wrong")
            self.assign_schedule(job, resources, time)
        self.queue_admission.clear()
