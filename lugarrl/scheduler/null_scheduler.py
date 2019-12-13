#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Any, Optional

from ..job import Job
from ..scheduler import Scheduler

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

    def __init__(self, number_of_processors, total_memory):
        self.current_slot: Optional[int] = None
        super().__init__(number_of_processors, total_memory)

    def step(self, offset: int = None) -> bool:
        "-1 (and, as a matter of fact, any negative number is a no-op.)"
        if self.current_slot is not None:
            raise AssertionError('current_slot invariant not true')

        self.current_slot = offset if offset is not None else -1
        return self.schedule()

    def forward_time(self):
        present = self.job_events.step(1)
        self.cluster = self.play_events(
            present, self.cluster, update_queues=True
        )
        self.current_time += 1
        self.schedule()

    @property
    def action_space(self):
        # We always support the null action
        return len(self.queue_admission) + 1

    def schedule(self) -> bool:
        try:
            if self.current_slot is not None \
               and len(self.queue_admission) \
               and 0 <= self.current_slot < len(self.queue_admission):
                job: Job = self.queue_admission[self.current_slot]
                if not self.cluster.fits(job):
                    return False
                resources = self.can_schedule_now(job)
                if resources:
                    self.assign_schedule(job, resources, self.current_time)
                    self.queue_admission.pop(self.current_slot)
                    return True
                else:
                    return False
            else:
                return False
        finally:
            self.current_slot = None

    def sjf_lt(self, a, b):
        return b is None or (a.requested_time < b.requested_time or
                                  (a.requested_time == b.requested_time and
                                   a.submission_time < b.submission_time))

    def sjf_action(self, limit: int) -> int:
        "Returns the index of the job SJF would pick."

        best = None
        bestidx = -1
        for i, job in enumerate(self.queue_admission[:limit]):
            if self.sjf_lt(job, best):
                if self.cluster.fits(job):
                    best = job
                    bestidx = i
        return bestidx
