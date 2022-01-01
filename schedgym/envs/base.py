#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from enum import IntEnum
from typing import List, Dict
from abc import ABC, abstractmethod

import gym

import numpy as np

from .simulator import SimulationType, DeepRmSimulator
from ..scheduler.null_scheduler import NullScheduler
from .workload import build as build_workload

BACKLOG_SIZE = 60
MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10

TIME_HORIZON = 20
JOB_SLOTS = 5
AMOUNT_OF_MEMORY = 10
NUMBER_OF_PROCESSORS = 10
MAXIMUM_JOB_LENGTH = 15
MAXIMUM_JOB_SIZE = 10
NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8
DEFAULT_WORKLOAD = {
    'type': 'deeprm',
    'new_job_rate': NEW_JOB_RATE,
    'max_job_size': MAXIMUM_JOB_SIZE,
    'max_job_len': MAXIMUM_JOB_LENGTH,
    'small_job_chance': SMALL_JOB_CHANCE,
}


class RewardJobs(IntEnum):
    ALL = (0,)
    JOB_SLOTS = (1,)
    WAITING = (2,)
    RUNNING_JOB_SLOTS = (3,)

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
    time_limit: int
    job_num_cap: int
    time_horizon: int
    ignore_memory: bool
    color_index: List[int]
    scheduler: NullScheduler
    color_cache: Dict[int, int]
    simulator: DeepRmSimulator

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

        self.time_horizon = kwargs.get(
            'time_horizon', TIME_HORIZON
        )  # number of time steps in the graph

        time_limit = kwargs.get('time_limit', 200)
        if time_limit is None:
            self.time_limit = 1
            self.update_time_limit = True
        else:
            self.time_limit = time_limit
            self.update_time_limit = False

        step = 1.0 / self.job_num_cap
        # zero is already present and set to "no job there"
        self.colormap = np.arange(start=step, stop=1, step=step)
        if self.shuffle_colors:
            np.random.shuffle(self.colormap)
        self.color_index = list(range(len(self.colormap)))

        # Number of jobs to show
        self.job_slots = kwargs.get('job_slots', JOB_SLOTS)

        self.reward_mapper = {
            RewardJobs.ALL: lambda: self.scheduler.jobs_in_system,
            RewardJobs.WAITING: lambda: self.scheduler.queue_admission,
            RewardJobs.JOB_SLOTS: lambda: self.scheduler.queue_admission[
                : self.job_slots
            ],
            RewardJobs.RUNNING_JOB_SLOTS: lambda: self.scheduler.queue_running
            + self.scheduler.queue_admission[: self.job_slots],
        }

        self.backlog_size = kwargs.get('backlog_size', BACKLOG_SIZE)
        self.memory = kwargs.get('memory', AMOUNT_OF_MEMORY)
        self.processors = kwargs.get('processors', NUMBER_OF_PROCESSORS)
        self.ignore_memory = kwargs.get('ignore_memory', False)

        self.workload_config = kwargs.get('workload', DEFAULT_WORKLOAD)
        wl = build_workload(self.workload_config)

        scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )
        self.simulator = DeepRmSimulator(
            wl,
            scheduler,
            simulation_type=self.simulation_type,
            job_slots=self.job_slots,
        )

    def reset(self) -> np.ndarray:
        scheduler = NullScheduler(
            self.processors, self.memory, ignore_memory=self.ignore_memory
        )
        wl = build_workload(self.workload_config)
        if self.update_time_limit and hasattr(wl, 'trace'):
            self.time_limit = (
                wl.trace[-1].submission_time +  # type: ignore
                wl.trace[-1].execution_time  # type: ignore
            )
        self.simulator.reset(wl, scheduler)
        return self.state

    def _render_state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
        s = self._convert_state(
            state,
            jobs,
            backlog,
            (
                (self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB
            ),
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
        memory = np.zeros(
            (self.job_slots, self.time_horizon, self.scheduler.total_memory)
        )
        processors = np.zeros(
            (
                self.job_slots,
                self.time_horizon,
                self.scheduler.number_of_processors,
            )
        )
        for i, j in enumerate(wait):
            if j.requested_processors == -1:
                break
            time_slice = slice(
                0,
                self.time_horizon
                if j.requested_time > self.time_horizon
                else j.requested_time,
            )
            processors[i, time_slice, : j.requested_processors] = 1.0
            if j.requested_memory != -1:
                memory[i, time_slice, : j.requested_memory] = 1.0
        return (processors,) if self.ignore_memory else (processors, memory)

    def _convert_state(self, current, wait, backlog, time):
        current = self.build_current_state(current)
        wait = self.build_job_slots(wait)
        backlog_width = self.backlog_size // self.time_horizon
        backlog = np.ones(self.time_horizon * backlog_width) * backlog
        unique = set(np.unique(current[0])) - {0.0}
        if len(unique) > self.job_num_cap:
            raise AssertionError('Number of jobs > number of colors')
        available_colors = list(
            set(self.color_index)
            - set(
                [self.color_cache[j] for j in unique if j in self.color_cache]
            )
        )
        need_color = unique - set(self.color_cache.keys())
        for i, j in enumerate(need_color):
            self.color_cache[j] = available_colors[i]
        for j in unique:  # noqa
            for resource in current:
                resource[resource == j] = self.colormap[self.color_cache[j]]

        return (
            np.array(current),
            np.array(wait),
            backlog.reshape((self.time_horizon, -1)),
            np.ones((self.time_horizon, 1)) * min(1.0, time),
        )

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

    def compute_reward(self, joblist):
        return -np.sum([1 / j.execution_time for j in joblist])

    @property
    def reward(self):
        return self.compute_reward(self.reward_mapper[self.reward_jobs]())

    @property
    def stats(self):
        return self.scheduler.stats

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    def scheduler(self):
        return self.simulator.scheduler
