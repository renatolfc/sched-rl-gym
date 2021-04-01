#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations, division

import gym
import numpy as np
from gym import utils, spaces

from .. import simulator
from .deeprm_env import DeepRmEnv
from ..scheduler.null_scheduler import NullScheduler
from .workload import build as build_workload, DeepRmWorkloadGenerator
from .workload import SyntheticWorkloadGenerator

from .base import DeepRmSimulator, BaseRmEnv

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

        self.time_horizon = kwargs.get('time_horizon', TIME_HORIZON)  # number of time steps in the graph

        self.workload_config = kwargs.get('workload', DEFAULT_WORKLOAD)

        self.job_slots = kwargs.get('job_slots', JOB_SLOTS)  # Number of jobs to show

        self.ignore_memory = kwargs.get('ignore_memory', False)

        self.renderer = kwargs.get('renderer', None)

        self._setup_spaces()

    def _setup_spaces(self):
        self.scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )

        self.action_space = spaces.discrete.Discrete(self.job_slots + 1)

        # TODO: add more cluster-related features
        # 2 * time_horizon
        # job_slots * (7)
        # backlog

        cluster, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
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
                1 / j.execution_time for j in self.scheduler.jobs_in_system
            ])

        done = self.scheduler.current_time > self.time_limit or done

        return self.state, reward, done, {} if not done else self.stats

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
        state = np.array([(e[0], e[1]) for e in state[0]]).reshape((-1,))
        jobs = np.array(jobs).reshape((-1,))
        backlog = backlog * np.ones(1)
        return np.hstack((state, jobs, backlog))

    def reset(self):
        self.scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )
        wl = build_workload(self.workload_config)
        self.simulator = DeepRmSimulator(wl, self.scheduler)

        return self.state

    @property
    def stats(self):
        return self.scheduler.stats
