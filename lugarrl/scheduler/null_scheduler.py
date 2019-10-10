#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, Any

from lugarrl.job import Job
from lugarrl.scheduler import Scheduler

# The main issue here is that we have two kinds of steps:
# 1. OpenAI Gym steps
# 2. Scheduler steps
# For OpenAI Gym steps, we need to pass an action. For Scheduler steps, we need to pass an offset.

class NullScheduler(Scheduler):
    """An scheduler that receives scheduling commands from a client.

    This is a null scheduler in the sense that scheduling decisions aren't made
    by this class, but by another class, which forwards its decisions to this
    one so that they can be propagated into the simulator. As such, this
    implements the interface between RL environments (such as OpenAI gym)
    and the scheduler simulator.
    """
    def __init__(self):
        self.current_slot = None  # The job slot we wish to operate on

    def step(self, slot: int) -> bool:
        "-1 (and, as a matter of fact, any negative number is a no-op.)"
        if self.current_slot is not None:
            raise AssertionError('current_slot invariant not true')

        self.current_slot = slot
        if self.queue_admission and 0 <= slot <= len(self.queue_admission):
            self.schedule()
        else:
            self.forward_time()

    def forward_time(self):
        present = self.job_events.step(1)
        self.cluster = self.play_events(present, self.cluster, update_queues=True)
        self.current_time += 1
        self.schedule()

    @property
    def action_space(self):
        # We always support the null action
        return len(self.queue_admission) + 1

    def schedule(self) -> None:
        try:
            if self.current_slot is not None \
               and 0 <= self.current_slot <= len(self.queue_admission):
                job: Job = self.queue_admission[self.current_slot]
                resources = self.can_schedule_now(job)
                if resources:
                    self.assign_schedule(job, resources, self.current_time)
                    self.queue_admission.pop(self.current_slot)
                else:
                    self.forward_time()  # Any invalid index forwards time
        finally:
            self.current_slot = None
