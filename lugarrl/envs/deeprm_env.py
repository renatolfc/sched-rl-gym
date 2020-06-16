#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations, division

import random
import itertools

from typing import Optional
from collections import namedtuple

import numpy as np

import gym
from gym import utils, spaces

from .render import DeepRmRenderer
from ..scheduler import null_scheduler as ns
from .. import job, workload as wl, simulator

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
MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10

NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8

JobParameters = namedtuple('JobParameters', ['small', 'large'])


class WorkloadGenerator(wl.DistributionalWorkloadGenerator):
    def __init__(self, *args: wl.BinomialWorkloadGenerator):
        self.generators = args
        self.counter = itertools.count(1)

        for generator in self.generators:
            generator.counter = self.counter

    def sample(self, submission_time=0) -> Optional[job.Job]:
        return self.generators[
            random.randint(0, len(self.generators) - 1)
        ].sample(submission_time)


class DeepRmSimulator(simulator.TimeBasedSimulator):
    last_job_time: int

    def __init__(self, workload_generator: wl.WorkloadGenerator,
                 scheduler: ns.NullScheduler):
        if not isinstance(workload_generator, WorkloadGenerator) \
                or not isinstance(scheduler, ns.NullScheduler):
            raise AssertionError("Invalid arguments received.")
        super().__init__(workload_generator, scheduler)
        self.last_job_time = 0

    def step(self, submit=True):
        raise NotImplementedError('This simulator cannot follow the base API')

    def rl_step(self, action: int) -> bool:
        "Returns True when time passes."
        if self.scheduler.step(action):
            return False
        else:
            self.current_time += 1
            job = self.workload.sample(self.current_time)
            if job:
                self.scheduler.submit(job)
                self.last_job_time = self.current_time
            self.scheduler.forward_time()
            return True


class DeepRmEnv(gym.Env, utils.EzPickle):
    n_work: int
    job_slots: int
    n_resources: int
    max_job_len: int
    job_num_cap: int
    time_horizon: int
    max_job_size: int
    backlog_size: int
    use_raw_sate: bool
    new_job_rate: float
    small_job_chance: float
    simulator: DeepRmSimulator
    scheduler: ns.NullScheduler
    workload: WorkloadGenerator
    observation_space: spaces.box.Tuple
    action_space: spaces.discrete.Discrete

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self.color_cache = {}

        self.renderer = kwargs.get('renderer', None)
        self.time_limit = kwargs.get('time_limit', 200)
        self.use_raw_state = kwargs.get('use_raw_state', False)

        self.memory = kwargs.get('memory', AMOUNT_OF_MEMORY)
        self.processors = kwargs.get('processors', NUMBER_OF_PROCESSORS)

        self.n_resources = kwargs.get('n_resources', NUMBER_OF_RESOURCES)  # resources in the system
        self.n_work = kwargs.get('n_work', MAXIMUM_QUEUE_SIZE)  # max amount of work in the queue
        self.time_horizon = kwargs.get('time_horizon', TIME_HORIZON)  # number of time steps in the graph

        self.max_job_len = kwargs.get('max_job_len', MAXIMUM_JOB_LENGTH)  # max duration of new jobs
        self.max_job_size = kwargs.get('max_job_size', MAXIMUM_JOB_SIZE)

        self.backlog_size = kwargs.get('backlog_size', BACKLOG_SIZE)  # backlog queue size
        self.job_num_cap = kwargs.get('job_num_cap', MAXIMUM_NUMBER_OF_ACTIVE_JOBS)

        self.new_job_rate = kwargs.get('new_job_rate', NEW_JOB_RATE)  # rate of job arrival
        self.small_job_chance = kwargs.get('small_job_chance', SMALL_JOB_CHANCE)  # chance a new job is small

        self.job_slots = kwargs.get('job_slots', JOB_SLOTS)  # Number of jobs to show

        if self.backlog_size % self.time_horizon:
            raise AssertionError('Backlog must be a multiple of time horizon')

        self.backlog_width = self.backlog_size // self.time_horizon

        self.action_space = spaces.discrete.Discrete(self.job_slots + 1)

        self.scheduler = ns.NullScheduler(
            self.processors, self.memory
        )

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

        step = 1.0 / self.job_num_cap
        # zero is already present and set to "no job there"
        self.colormap = np.arange(start=step, stop=1, step=step)
        self.color_index = list(range(len(self.colormap)))

        utils.EzPickle.__init__(self, **kwargs)

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots, self.backlog_size
        )
        s = self._convert_state(
            state[0], state[1], jobs[0], jobs[1], backlog,
            ((self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB)
        )
        if self.use_raw_state:
            return s
        return self.pack_observation(s)

    def pack_observation(self, ob):
        ob = list(ob)
        ob[2] = np.hstack(ob[2]).reshape((self.time_horizon, -1))
        ob[3] = np.hstack(ob[3]).reshape((self.time_horizon, -1))
        return np.hstack(ob)

    def _convert_state(self, procs, mem, wait_procs, wait_mem, backlog, time):
        unique = set(np.unique(procs)) - set([0.0])
        if len(unique) > self.job_num_cap:
            raise AssertionError("Number of jobs > number of colors")
        available_colors = list(set(self.color_index) - set(
            [self.color_cache[j] for j in unique if j in self.color_cache]
        ))
        need_color = unique - set(self.color_cache.keys())
        for i, j in enumerate(need_color):
            self.color_cache[j] = available_colors[i]
        for job in unique:  # noqa
            procs[procs == job] = self.colormap[self.color_cache[job]]
            mem[mem == job] = self.colormap[self.color_cache[job]]
        wait_procs[wait_procs != 0] = 1.0
        wait_mem[wait_mem != 0] = 1.0

        return procs, mem, wait_procs, wait_mem, \
            backlog.reshape((self.time_horizon, -1)), \
            np.ones((self.time_horizon, 1)) * min(1.0, time)

    def find_slot_position(self, action):
        for i, j in enumerate(self.scheduler.queue_admission):
            if j.slot_position == action:
                return i
        return self.action_space.n - 1

    def step(self, action: int):
        if 0 <= action < self.action_space.n - 1:
            action = self.find_slot_position(action)
        time_passed = self.simulator.rl_step(action)

        reward = 0
        if time_passed:
            reward = -np.sum([
                1 / j.execution_time for j in self.scheduler.jobs_in_system
            ])

        done = self.scheduler.current_time > self.time_limit

        return self.state, reward, done, {}

    def reset(self):
        self.scheduler = ns.NullScheduler(
            self.processors, self.memory
        )
        workload = self._build_workload_generator()

        self.simulator = DeepRmSimulator(workload, self.scheduler)

        return self.state

    def render(self, mode='human'):
        if self.renderer is None:
            self.renderer = DeepRmRenderer(mode)
        if self.use_raw_state:
            rgb, size = self.renderer.render(self.state)
        else:
            self.use_raw_state = True
            rgb, size = self.renderer.render(self.state)
            self.use_raw_state = False
        return np.frombuffer(rgb, dtype=np.uint8).reshape(
            (size[0], size[1], 3)
        )

    def _build_workload_generator(self):
        # Time-related job parameters {{{
        self.small_job_time_lower = 1
        self.small_job_time_upper = max(self.max_job_len // 5, 1)
        self.large_job_time_lower = int(self.max_job_len * (2 / 3))
        self.large_job_time_upper = self.max_job_len
        # }}}

        # Resource-related job parameters {{{
        self.dominant_resource_lower = self.max_job_size // 2
        self.dominant_resource_upper = self.max_job_size
        self.other_resource_lower = 1
        self.other_resource_upper = self.max_job_size // 5
        # }}}

        cpu_dominant_parameters = JobParameters(  # {{{
            job.JobParameters(
                self.small_job_time_lower,
                self.small_job_time_upper,
                self.dominant_resource_lower,
                self.dominant_resource_upper,
                self.other_resource_lower,
                self.other_resource_upper
            ),
            job.JobParameters(
                self.large_job_time_lower,
                self.large_job_time_upper,
                self.dominant_resource_lower,
                self.dominant_resource_upper,
                self.other_resource_lower,
                self.other_resource_upper
            ),
        )  # }}}

        mem_dominant_parameters = JobParameters(  # {{{
            job.JobParameters(
                self.small_job_time_lower,
                self.small_job_time_upper,
                self.other_resource_lower,
                self.other_resource_upper,
                self.dominant_resource_lower,
                self.dominant_resource_upper,
            ),
            job.JobParameters(
                self.large_job_time_lower,
                self.large_job_time_upper,
                self.other_resource_lower,
                self.other_resource_upper,
                self.dominant_resource_lower,
                self.dominant_resource_upper,
            ),
        )  # }}}

        return WorkloadGenerator(
            wl.BinomialWorkloadGenerator(
                self.new_job_rate, self.small_job_chance,
                cpu_dominant_parameters.small, cpu_dominant_parameters.large
            ),
            wl.BinomialWorkloadGenerator(
                self.new_job_rate, self.small_job_chance,
                mem_dominant_parameters.small, mem_dominant_parameters.large
            ),
        )

    @property
    def slowdown(self):
        return self.scheduler.slowdown

