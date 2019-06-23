#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from lugarrl.scheduler import Scheduler


class RandomScheduler(Scheduler):
    def schedule(self) -> None:
        while any([self.can_schedule_now(j) for j in self.queue_admission]):
            job = self.queue_admission[random.randint(0, len(self.queue_admission))]
            resources = self.can_schedule_now(job)
            if resources:
                self.assign_schedule(job, resources, self.current_time)
                self.queue_admission.remove(job)
