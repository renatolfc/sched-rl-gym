#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from abc import ABC, abstractmethod

import gym
from gym import utils, spaces

import numpy as np

from .. import simulator
from ..scheduler.null_scheduler import NullScheduler
from .workload import DeepRmWorkloadGenerator
from .workload import SyntheticWorkloadGenerator

MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10


class DeepRmSimulator(simulator.TimeBasedSimulator):
    last_job_time: int
    scheduler: NullScheduler

    def __init__(self, workload_generator: DeepRmWorkloadGenerator,
                 scheduler: NullScheduler):
        if (not isinstance(workload_generator, DeepRmWorkloadGenerator)
                and not isinstance(
                    workload_generator, SyntheticWorkloadGenerator
                )) \
                or not isinstance(scheduler, NullScheduler):
            raise AssertionError("Invalid arguments received.")
        super().__init__(workload_generator, scheduler)

        self.last_job_time = 0
        if isinstance(workload_generator, SyntheticWorkloadGenerator):
            first_job_time = workload_generator.peek().submission_time - 1
            workload_generator.current_time = first_job_time
            scheduler.job_events.time = first_job_time
            scheduler.current_time = first_job_time

    def step(self, submit=True):
        "Not implemented in DeepRmSimulator"
        raise NotImplementedError('This simulator cannot follow the base API')

    def rl_step(self, action: int) -> bool:
        "Returns True when time passes."
        if self.scheduler.step(action):
            return False
        else:
            self.current_time += 1
            j = self.workload.step()
            if j:
                self.scheduler.submit(j)
                self.last_job_time = self.current_time
            self.scheduler.forward_time()
            return True


class BaseRmEnv(ABC, gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    job_slots: int
    job_num_cap: int
    time_horizon: int
    ignore_memory: bool
    color_index: List[int]
    scheduler: NullScheduler
    color_cache: dict[int, float]

    @abstractmethod
    def __init__(self, **kwargs):
        self.color_cache = {}
        self.renderer = kwargs.get('renderer', None)
        self.shuffle_colors = kwargs.get('shuffle_colors', False)
        self.job_num_cap = kwargs.get('job_num_cap', MAXIMUM_NUMBER_OF_ACTIVE_JOBS)


        step = 1.0 / self.job_num_cap
        # zero is already present and set to "no job there"
        self.colormap = np.arange(start=step, stop=1, step=step)
        if self.shuffle_colors:
            np.random.shuffle(self.colormap)
        self.color_index = list(range(len(self.colormap)))


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
        backlog = np.ones(self.time_horizon) * backlog
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

