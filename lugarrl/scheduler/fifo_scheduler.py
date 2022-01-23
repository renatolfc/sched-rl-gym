#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lugarrl.scheduler import Scheduler


class FifoScheduler(Scheduler):
    def schedule(self) -> None:
        for job in [j for j in self.queue_admission]:
            time = self.find_first_time_for(job)
            if self.queue_waiting and time < max([j.start_time for j in self.queue_waiting]):
                continue
            self.assign_schedule(job, time)
            self.queue_admission.remove(job)
