#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations, division

import gym
import numpy as np
from gym import utils, spaces

from ..scheduler.null_scheduler import NullScheduler
from .workload import build as build_workload
from ..job import JobState

from .base import BaseRmEnv
from .simulator import DeepRmSimulator

import logging
logger = logging.getLogger(__name__)


TIME_HORIZON = 20
MAXIMUM_JOB_LENGTH = 15
RESOURCE_SLOTS = 10
MAXIMUM_JOB_SIZE = 10
BACKLOG_SIZE = 60
JOB_SLOTS = 5

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

        self.time_limit = kwargs.get('time_limit', 200)

        self.memory = kwargs.get('memory', AMOUNT_OF_MEMORY)
        self.processors = kwargs.get('processors', NUMBER_OF_PROCESSORS)

        # number of time steps in the graph
        self.time_horizon = kwargs.get('time_horizon', TIME_HORIZON)

        self.workload_config = kwargs.get('workload', DEFAULT_WORKLOAD)

        # Number of jobs to show
        self.job_slots = kwargs.get('job_slots', JOB_SLOTS)
        # backlog queue size
        self.backlog_size = kwargs.get('backlog_size', BACKLOG_SIZE)

        self.ignore_memory = kwargs.get('ignore_memory', False)

        self.renderer = kwargs.get('renderer', None)

        self.maximum_work = self.time_limit * self.processors
        self.maximum_work_mem = self.time_limit * self.memory

        self.scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )

        wl = build_workload(self.workload_config)
        self.simulator = DeepRmSimulator(
            wl, self.scheduler, simulation_type=self.simulation_type,
            job_slots=self.job_slots
        )

        self._setup_spaces()

    def _setup_spaces(self):
        self.action_space = spaces.discrete.Discrete(self.job_slots + 1)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=((len(self.state),)),
            dtype=np.float32
        )

    def step(self, action: int):
        done = False
        if not (0 <= action < self.action_space.n - 1):
            action = None

        try:
            time_passed = self.simulator.rl_step(action)
        except StopIteration:
            time_passed = True
            done = True

        reward = 0
        if time_passed:
            reward = -np.sum([
                1 / j.execution_time for j in
                self.scheduler.queue_admission[:self.job_slots]
            ])

        done = self.scheduler.current_time > self.time_limit or done

        return self.state, reward, done, {} if not done else self.stats

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
        newstate = np.zeros((
            len(state[0]) * (1 if self.ignore_memory else 2) * 2
        ))
        newstate[:len(state[0]) * 2] = np.array(
            [(e[0], e[1]) for e in state[0]]
        ).reshape((-1,)) / self.processors
        if not self.ignore_memory:
            newstate[len(state[0]) * 2:] = np.array(
                [(e[0], e[1]) for e in state[1]]
            ).reshape((-1,)) / self.memory
        jobs = self._normalize_jobs(jobs).reshape((-1,))
        backlog = backlog * np.ones(1) / BACKLOG_SIZE

        running = [j for j in self.scheduler.queue_running
                   if j.submission_time + j.requested_time>
                   self.scheduler.current_time]

        remaining_work = sum([
            (j.submission_time + j.requested_time -
             self.scheduler.current_time) *
            j.requested_processors
            for j in running
        ]) / self.maximum_work
        remaining_work_mem = sum([
            (j.submission_time + j.requested_time -
             self.scheduler.current_time) *
            j.requested_memory
            for j in running
        ]) / self.maximum_work_mem

        # XXX: this normalization only works while we're sampling at most one
        # job per time step. Once this is not true, we risk having the
        # queue_size feature > 1.0 (which is incorrect)
        queue_size = len(self.scheduler.queue_admission) / self.time_limit
        time_left = 1 - self.scheduler.current_time / self.time_limit

        try:
            next_free = min(
                running,
                key=lambda x: x.start_time + x.requested_time
            )
            next_free = np.array((
                (next_free.start_time + next_free.requested_time -
                 self.scheduler.current_time) / self.time_limit,
                next_free.requested_processors / self.processors,
                (state[0][0][0] + next_free.requested_processors) /
                self.processors
            ))
        except ValueError:
            next_free = np.array((0, 0, 1.0))

        return np.hstack((
            newstate, jobs, backlog, next_free,
            np.array((
                remaining_work, remaining_work_mem, queue_size, time_left
            ))
        ))

    def _normalize_jobs(self, jobs):
        def _sumdiv(arr, idx, orig, limit):
            arr[idx] = (orig + 1) / (limit + 1)

        ret = np.zeros((len(jobs), len(jobs[0])))
        for i, job in enumerate(jobs):
            _sumdiv(ret[i], 0, job.submission_time, self.time_limit)
            _sumdiv(ret[i], 1, job.requested_time, self.time_limit)
            _sumdiv(ret[i], 2, job.requested_memory, self.memory)
            _sumdiv(ret[i], 3, job.requested_processors, self.processors)
            _sumdiv(ret[i], 4, job.queue_size, self.time_limit)
            _sumdiv(ret[i], 5, job.queued_work,
                    self.time_limit * self.time_limit * self.processors)
            _sumdiv(ret[i], 6, job.free_processors, self.processors)
        return ret

    def reset(self):
        self.scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )
        wl = build_workload(self.workload_config)
        self.simulator.reset(wl, self.scheduler)

        return self.state

    @property
    def stats(self):
        return self.scheduler.stats
