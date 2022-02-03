#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations, division

import numpy as np
import gym.spaces.box
import gym.spaces.discrete

from ..job import Job
from .base import BaseRmEnv

import logging

logger = logging.getLogger(__name__)


MAXIMUM_JOB_LENGTH = 15
RESOURCE_SLOTS = 10
MAXIMUM_JOB_SIZE = 10

AMOUNT_OF_MEMORY = 10
NUMBER_OF_RESOURCES = 2
NUMBER_OF_PROCESSORS = 10
MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10

NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8

DEFAULT_WORKLOAD = {
    'type': 'deeprm',
    'new_job_rate': NEW_JOB_RATE,
    'max_job_size': MAXIMUM_JOB_SIZE,
    'max_job_len': MAXIMUM_JOB_LENGTH,
    'small_job_chance': SMALL_JOB_CHANCE,
}


class CompactRmEnv(BaseRmEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.memory = kwargs.get('memory', AMOUNT_OF_MEMORY)
        self.processors = kwargs.get('processors', NUMBER_OF_PROCESSORS)

        self.renderer = kwargs.get('renderer', None)

        self.maximum_work = self.processors
        self.maximum_work_mem = self.memory

        self._setup_spaces()

    def _setup_spaces(self):
        self.action_space = gym.spaces.discrete.Discrete(self.job_slots + 1)

        self.observation_space = gym.spaces.box.Box(
            low=0.0, high=1.0, shape=((len(self.state),)), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        super().reset()
        self.maximum_work = self.time_limit * self.processors
        self.maximum_work_mem = self.time_limit * self.memory
        return super().reset()

    def step(self, action: int):
        done = False
        found = True
        if not (0 <= action < self.action_space.n - 1):
            found = False

        try:
            intermediate = self.simulator.rl_step(
                action if found else None, self.reward_mapper[self.reward_jobs]
            )
            # XXX: This is technically incorrect. The correct thing to do here
            # is: when we have a trace-based workload generator, we need to
            # maintain a check on whether we want to sample from it or not, and
            # use the time limit to actually decide whether we're done or not.
            # In the current setting, we might potentially "lose" the last jobs
            # of the workload.
        except StopIteration:
            intermediate = [[Job()]]
            done = True

        reward = self.reward if any(intermediate) else 0
        done = bool(self.time_limit) and (
            self.scheduler.current_time > self.time_limit or done
        )

        if not done and self.smdp and any(intermediate):
            rewards = [self.compute_reward(js) for js in intermediate]
            rewards[0] = 0
            reward = (
                self.gamma ** np.arange(len(intermediate))
            ).dot(rewards)

        return (
            self.state,
            reward,
            done,
            self.stats if done else None
        )

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots, self.smdp
        )
        snapshots = len(state[0])
        newstate = np.zeros(
            snapshots + snapshots * (1 if self.ignore_memory else 2) * 2
        )

        stateslice = slice(snapshots * (2 if self.ignore_memory else 4), None)
        newstate[stateslice] = state[0]
        newstate[stateslice] /= self.time_limit

        newstate[: snapshots * 2] = (
            np.array(
                [(e[0], e[1]) for e in state[1]],
                dtype=np.float32
            ).reshape((-1,),) / self.processors
        )
        if not self.ignore_memory:
            newstate[snapshots * 2:snapshots * 4] = (
                np.array(
                    [(e[0], e[1]) for e in state[2]],
                    dtype=np.float32
                ).reshape((-1,)) / self.memory
            )
        jobs = self._normalize_jobs(jobs).reshape((-1,))
        backlog = backlog * np.ones(1) / self.backlog_size

        running = [
            j
            for j in self.scheduler.queue_running
            if j.submission_time + j.requested_time
            > self.scheduler.current_time
        ]

        remaining_work = (
            sum(
                [
                    (
                        j.submission_time
                        + j.requested_time
                        - self.scheduler.current_time
                    )
                    * j.requested_processors
                    for j in running
                ]
            )
            / self.maximum_work
        )
        remaining_work_mem = (
            sum(
                [
                    (
                        j.submission_time
                        + j.requested_time
                        - self.scheduler.current_time
                    )
                    * j.requested_memory
                    for j in running
                ]
            )
            / self.maximum_work_mem
        )

        # XXX: this normalization only works while we're sampling at most one
        # job per time step. Once this is not true, we risk having the
        # queue_size feature > 1.0 (which is incorrect)
        queue_size = len(self.scheduler.queue_admission) / self.time_limit
        time_left = 1 - self.scheduler.current_time / self.time_limit

        try:
            next_free = min(
                running, key=lambda x: x.start_time + x.requested_time
            )
            next_free = np.array(
                (
                    (
                        next_free.start_time
                        + next_free.requested_time
                        - self.scheduler.current_time
                    )
                    / self.time_limit,
                    next_free.requested_processors / self.processors,
                    (state[1][0][0] + next_free.requested_processors)
                    / self.processors,
                )
            )
        except ValueError:
            next_free = np.array((0, 0, 1.0))

        return np.hstack(
            (
                newstate,
                jobs,
                backlog,
                next_free,
                np.array(
                    (remaining_work, remaining_work_mem, queue_size, time_left)
                ),
            ),
        )

    def _normalize_jobs(self, jobs):
        def _sumdiv(arr, idx, orig, limit):
            arr[idx] = (orig + 1) / (limit + 1)

        ret = np.zeros((len(jobs), len(jobs[0])), dtype=np.float32)
        for i, job in enumerate(jobs):
            _sumdiv(ret[i], 0, job.submission_time, self.time_limit)
            _sumdiv(ret[i], 1, job.requested_time, self.time_limit)
            _sumdiv(ret[i], 2, job.requested_memory, self.memory)
            _sumdiv(ret[i], 3, job.requested_processors, self.processors)
            _sumdiv(ret[i], 4, job.queue_size, self.time_limit)
            _sumdiv(
                ret[i],
                5,
                job.queued_work,
                self.time_limit * self.time_limit * self.processors,
            )
            _sumdiv(ret[i], 6, job.free_processors, self.processors)
        return ret
