#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations, division

from typing import Union

import numpy as np

import gym
from gym import utils, spaces

from .base import BaseRmEnv
from .simulator import DeepRmSimulator
from ..scheduler.null_scheduler import NullScheduler
from .workload import build as build_workload, DeepRmWorkloadGenerator

import logging
logger = logging.getLogger(__name__)

MAXIMUM_QUEUE_SIZE = 16

TIME_HORIZON = 20
MAXIMUM_JOB_LENGTH = 15
RESOURCE_SLOTS = 10
MAXIMUM_JOB_SIZE = 10
BACKLOG_SIZE = 60
JOB_SLOTS = 5

AMOUNT_OF_MEMORY = 10
NUMBER_OF_RESOURCES = 2
NUMBER_OF_PROCESSORS = 10
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


class DeepRmEnv(BaseRmEnv):
    n_work: int
    job_slots: int
    n_resources: int
    time_horizon: int
    backlog_size: int
    use_raw_sate: bool
    simulator: DeepRmSimulator
    scheduler: NullScheduler
    workload: DeepRmWorkloadGenerator
    observation_space: Union[spaces.tuple.Tuple, spaces.box.Box]
    action_space: spaces.discrete.Discrete

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.time_limit = kwargs.get('time_limit', 200)
        self.use_raw_state = kwargs.get('use_raw_state', False)

        self.memory = kwargs.get('memory', AMOUNT_OF_MEMORY)
        self.processors = kwargs.get('processors', NUMBER_OF_PROCESSORS)

        self.n_resources = kwargs.get('n_resources', NUMBER_OF_RESOURCES)  # resources in the system
        self.n_work = kwargs.get('n_work', MAXIMUM_QUEUE_SIZE)  # max amount of work in the queue
        self.time_horizon = kwargs.get('time_horizon', TIME_HORIZON)  # number of time steps in the graph

        self.backlog_size = kwargs.get('backlog_size', BACKLOG_SIZE)  # backlog queue size

        self.workload_config = kwargs.get('workload', DEFAULT_WORKLOAD)

        self.job_slots = kwargs.get('job_slots', JOB_SLOTS)  # Number of jobs to show

        self.ignore_memory = kwargs.get('ignore_memory', False)

        if self.backlog_size % self.time_horizon:
            raise AssertionError('Backlog must be a multiple of time horizon')

        self.backlog_width = self.backlog_size // self.time_horizon

        self.simulator = DeepRmSimulator(
            build_workload(self.workload_config),
            NullScheduler(
                self.processors, self.memory, ignore_memory=self.ignore_memory
            ),
            job_slots=self.job_slots
        )

        self.setup_spaces()

    @property
    def scheduler(self):
        return self.simulator.scheduler

    def setup_spaces(self):
        self.action_space = spaces.discrete.Discrete(self.job_slots + 1)
        if self.use_raw_state:
            self.setup_raw_spaces()
        else:
            self.setup_image_spaces()

    def setup_image_spaces(self):
        self.observation_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.time_horizon,
                ((0 if self.ignore_memory else (self.job_slots + 1))
                    * self.scheduler.total_memory) +
                (self.job_slots + 1) * self.scheduler.number_of_processors +
                self.backlog_width +
                1
            )
        )

    def setup_raw_spaces(self):
        self.memory_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.time_horizon, self.scheduler.total_memory
            )
        )
        self.processor_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.time_horizon, self.scheduler.number_of_processors
            )
        )
        self.backlog_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.time_horizon, self.backlog_width
            )
        )
        self.memory_slots_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.job_slots, self.time_horizon, self.scheduler.total_memory
            )
        )
        self.processor_slots_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.job_slots, self.time_horizon,
                self.scheduler.number_of_processors
            )
        )
        self.time_since_space = spaces.Discrete(self.time_horizon)

        self.observation_space = spaces.tuple.Tuple((
            self.processor_space, self.memory_space,
            self.processor_slots_space, self.memory_slots_space,
            self.backlog_space, self.time_since_space
        ))
        self.observation_space.n = np.sum([
            np.prod(e.shape) if isinstance(e, spaces.box.Box) else e.n
            for e in self.observation_space
        ])

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
        s = self._convert_state(
            state, jobs, backlog,
            ((self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB)
        )
        if self.use_raw_state:
            return s
        return self.pack_observation(s)

    def pack_observation(self, ob):
        current, wait, backlog, time = ob
        wait = wait.reshape(self.time_horizon, -1)
        current = current.reshape(self.time_horizon, -1)
        return np.hstack((current, wait, backlog, time))

    def find_slot_position(self, action):
        if action < len(self.scheduler.queue_admission):
            return action
        return self.action_space.n - 1

    def step(self, action: int):
        done = False
        if 0 <= action < self.action_space.n - 1:
            action = self.find_slot_position(action)
        else:
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

        return self.state, reward, done, {}

    def reset(self):
        scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )

        wl = build_workload(self.workload_config)
        self.simulator.reset(wl, scheduler)
        return self.state

    @property
    def slowdown(self):
        slowdown = self.scheduler.slowdown
        if slowdown:
            return slowdown
        # no job has finished yet, return maximum slowdown possible
        exec_time = MAXIMUM_JOB_LENGTH
        return [(exec_time + self.time_limit) / exec_time]

    @property
    def stats(self):
        return self.scheduler.stats
