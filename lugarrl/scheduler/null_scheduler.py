#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""null_scheduler - a module that doesn't do any scheduling

The purposed of this module is to provide a way for clients of the simulator to
implement different scheduling strategies.

Most notably, this can be used by learning agents to select which jobs to
schedule in an iterative way.
"""

from typing import Optional

from ..job import Job
from ..scheduler import Scheduler

# The main issue here is that we have two kinds of steps:
# 1. OpenAI Gym steps
# 2. Scheduler steps
# For OpenAI Gym steps, we need to pass an action. For Scheduler steps, we need
# to pass an offset.


class NullScheduler(Scheduler):
    """An scheduler that receives scheduling commands from a client.

    This is a null scheduler in the sense that scheduling decisions aren't made
    by this class, but by another class, which forwards its decisions to this
    one so that they can be propagated into the simulator. As such, this
    implements the interface between RL environments (such as OpenAI gym)
    and the scheduler simulator.

    Parameters
    ----------
        number_of_processors : int
            The number of processors managed by this scheduler
        total_memory : int
            The total amount of memory in the cluster managed by this scheduler
    """

    current_slot: Optional[int]

    def __init__(self, number_of_processors, total_memory, ignore_memory=False):
        self.current_slot: Optional[int] = None
        super().__init__(number_of_processors, total_memory,
                         ignore_memory=ignore_memory)

    def step(self, offset: int = None) -> bool:
        """Steps the scheduler by setting which job to choose.

        Uses the offset to select a position in the admission queue. If the
        agent select a job that doesn't fit the cluster, such a selection is
        ignored by the scheduler.

        Differently from its base class, this method **does not** forward time.
        For this, please see :func:`forward_time`.

        Parameters
        ----------
            offset : int
                The offset in the admission queue of the job to select. Any
                negative number represents a no-op.
        """
        if self.current_slot is not None:
            raise AssertionError('current_slot invariant not true')

        self.current_slot = offset if offset is not None else -1
        return self.schedule()

    def forward_time(self):
        """Forwards time by one time step.

        For details, see :func:`step`.
        """

        present = self.job_events.step(1)
        self.cluster = self.play_events(
            present, self.cluster, update_queues=True
        )
        self.current_time += 1
        self.schedule()

    @property
    def action_space(self):
        """Helper that gives the number of actions available for the agent."""
        # We always support the null action
        return len(self.queue_admission) + 1

    def schedule(self) -> bool:
        """Tries to schedule the job selected with :func:`step`.

        When :func:`step` is called, it stores the job currently selected by
        the client. This function will check in the queue which job the
        selection corresponds to and will check if the job fits in the cluster
        *right now*.  If it does, the job is scheduled, otherwise, it is
        ignored.
        In either case, the current selection is cleared.

        Returns:
            bool: True if the selected job was scheduled. False otherwise.
        """
        try:
            if self.current_slot is not None \
               and len(self.queue_admission) > 0 \
               and 0 <= self.current_slot < len(self.queue_admission):
                job: Job = self.queue_admission[self.current_slot]
                if not self.cluster.fits(job):
                    return False
                resources = self.can_schedule_now(job)
                if resources:
                    self.assign_schedule(job, resources, self.current_time)
                    self.queue_admission.pop(self.current_slot)
                    return True
                return False
            return False
        finally:
            self.current_slot = None

    def sjf_lt(self, a: Job, b: Job):  # pylint: disable=C, no-self-use
        """Comparison function that gives the same ordering SJF would give.

        Parameters
        ----------
            a: Job
                A first job
            b: Job
                A second job

        Returns:
            bool: True if `a` is shorter than `b`. False otherwise.
        """
        return b is None or (a.requested_time < b.requested_time)

    def sjf_action(self, limit: int) -> int:
        """Returns the index of the job SJF would pick.

        Parameters
        ----------
            limit : int
                How far in the admission queue to look when searching for the
                shortest job.
        """

        best = None
        bestidx = limit
        for i, job in enumerate(self.queue_admission[:limit]):
            if self.sjf_lt(job, best):
                if self.cluster.fits(job):
                    best = job
                    bestidx = i
        return bestidx
