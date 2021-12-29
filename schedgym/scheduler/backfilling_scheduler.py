#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""backfilling_scheduler - Module for a conservative backfilling scheduler"""

from schedgym.scheduler import Scheduler


class BackfillingScheduler(Scheduler):
    """Implements a conservative backfilling scheduler."""

    def schedule(self) -> None:
        "Schedules a job according to the conservative backfilling strategy."
        for job in self.queue_admission:
            time, resources = self.find_first_time_for(job)
            if not resources:
                raise AssertionError('Something is terribly wrong')
            self.assign_schedule(job, resources, time)
        self.queue_admission.clear()
