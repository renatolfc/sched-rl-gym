#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from enum import IntEnum
from typing import List, Dict
from abc import ABC, abstractmethod

import gym

import numpy as np

from .simulator import SimulationType
from ..scheduler.null_scheduler import NullScheduler

MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10


class RewardJobs(IntEnum):
    ALL = 0,
    JOB_SLOTS = 1,
    WAITING = 2

    @staticmethod
    def from_str(reward_range: str):
        reward_range = reward_range.upper().replace('-', '_')
        if reward_range in RewardJobs.__members__:
            return RewardJobs[reward_range]
        else:
            raise ValueError(
                f'{reward_range} is not a valid RewardJobs range. '
                f'Valid options are: {list(RewardJobs.__members__.keys())}.'
            )


class BaseRmEnv(ABC, gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    job_slots: int
    job_num_cap: int
    time_horizon: int
    ignore_memory: bool
    color_index: List[int]
    scheduler: NullScheduler
    color_cache: Dict[int, float]

    @abstractmethod
    def __init__(self, **kwargs):
        self.color_cache = {}
        self.renderer = kwargs.get('renderer', None)
        self.shuffle_colors = kwargs.get('shuffle_colors', False)
        self.job_num_cap = kwargs.get(
            'job_num_cap', MAXIMUM_NUMBER_OF_ACTIVE_JOBS
        )
        self.simulation_type = SimulationType.from_str(
            kwargs.get('simulation_type', 'time_based')
        )

        self.reward_jobs = RewardJobs.from_str(
            kwargs.get('reward_jobs', 'all')
        )

        step = 1.0 / self.job_num_cap
        # zero is already present and set to "no job there"
        self.colormap = np.arange(start=step, stop=1, step=step)
        if self.shuffle_colors:
            np.random.shuffle(self.colormap)
        self.color_index = list(range(len(self.colormap)))

        self.reward_mapper = {
            RewardJobs.ALL: lambda: self.scheduler.jobs_in_system,
            RewardJobs.WAITING: lambda: self.scheduler.queue_admission,
            RewardJobs.JOB_SLOTS: lambda: self.scheduler.queue_admission[
                :self.job_slots
            ]
        }

    def _render_state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
        s = self._convert_state(
            state, jobs, backlog,
            ((self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB)
        )
        return s

    def build_current_state(self, current):
        ret = [np.zeros((self.time_horizon, sum(e[0][:-1]))) for e in current]
        for i, _ in enumerate(current):
            for t in range(self.time_horizon):
                for k, v in current[i][t][-1].items():
                    ret[i][t][slice(*k)] = v
        return ret

    def build_job_slots(self, wait):
        memory = np.zeros((
            self.job_slots, self.time_horizon, self.scheduler.total_memory
        ))
        processors = np.zeros((
            self.job_slots, self.time_horizon,
            self.scheduler.number_of_processors
        ))
        for i, j in enumerate(wait):
            if j.requested_processors == -1:
                break
            time_slice = slice(
                0,
                self.time_horizon if j.requested_time > self.time_horizon
                else j.requested_time,
            )
            processors[i, time_slice, :j.requested_processors] = 1.0
            if j.requested_memory != -1:
                memory[i, time_slice, :j.requested_memory] = 1.0
        return (processors,) if self.ignore_memory else (processors, memory)

    def _convert_state(self, current, wait, backlog, time):
        current = self.build_current_state(current)
        wait = self.build_job_slots(wait)
        backlog_width = self.backlog_size // self.time_horizon
        backlog = np.ones(self.time_horizon * backlog_width) * backlog
        unique = set(np.unique(current[0])) - {0.0}
        if len(unique) > self.job_num_cap:
            raise AssertionError("Number of jobs > number of colors")
        available_colors = list(set(self.color_index) - set(
            [self.color_cache[j] for j in unique if j in self.color_cache]
        ))
        need_color = unique - set(self.color_cache.keys())
        for i, j in enumerate(need_color):
            self.color_cache[j] = available_colors[i]
        for j in unique:  # noqa
            for resource in current:
                resource[resource == j] = self.colormap[self.color_cache[j]]

        return np.array(current), np.array(wait), \
            backlog.reshape((self.time_horizon, -1)), \
            np.ones((self.time_horizon, 1)) * min(1.0, time)

    def render(self, mode='human'):
        if self.renderer is None:
            from .render import DeepRmRenderer
            self.renderer = DeepRmRenderer(mode)
        rgb = self.renderer.render(self._render_state())
        return rgb

    def seed(self, seed=None):
        if seed is None:
            seed = random.randint(0, 99999999)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    @property
    def reward(self):
        return -np.sum([
            1 / j.execution_time for j in self.reward_mapper[self.reward_jobs]()
        ])
