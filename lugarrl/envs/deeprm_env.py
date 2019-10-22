#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations, division

import random
import itertools

from typing import Optional
from collections import namedtuple

import numpy as np

import gym
from gym import utils
from gym import error, spaces
from gym.utils import seeding

from .. import job, workload, simulator
from ..scheduler import null_scheduler as ns

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

NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8

JobParameters = namedtuple('JobParameters', ['small', 'large'])


class WorkloadGenerator(workload.DistributionalWorkloadGenerator):
    def __init__(self, *args: workload.BinomialWorkloadGenerator):
        self.generators = args
        self.counter = itertools.count()

        for generator in self.generators:
            generator.counter = self.counter

    def sample(self, submission_time=0) -> Optional[job.Job]:
        return self.generators[
            random.randint(0, len(self.generators) - 1)
        ].sample(submission_time)


class DeepRmSimulator(simulator.TimeBasedSimulator):
    def __init__(self, workload_generator: workload.WorkloadGenerator,
                 scheduler: ns.NullScheduler):
        if not isinstance(workload_generator, WorkloadGenerator) \
                or not isinstance(scheduler, ns.NullScheduler):
            raise AssertionError("Invalid arguments received.")
        super().__init__(workload_generator, scheduler)

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
    new_job_rate: float
    small_job_chance: float
    simulator: DeepRmSimulator
    scheduler: ns.NullScheduler
    workload: WorkloadGenerator
    observation_space: spaces.box.Box
    action_space: spaces.discrete.Discrete

    metadata = {'render.modes': 'human'}

    def __init__(self):
        self._configure_environment()

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots, self.backlog_size,
#            self.job_num_cap
        )
        return state[0], state[1], jobs[0], jobs[1], backlog

    def step(self, action: int):
        time_passed = self.simulator.rl_step(action)

        reward = 0
        if time_passed:
            reward = -np.sum([
                1 / j.execution_time for j in self.scheduler.jobs_in_system
            ])

        return self.state, reward, False, {}

    def reset(self):
        self.scheduler = ns.NullScheduler(
            self.processors, self.memory
        )
        workload = self._build_workload_generator()

        self.simulator = DeepRmSimulator(workload, self.scheduler)

        if self.backlog_size % self.time_horizon:
            raise AssertionError('Backlog must be a multiple of time horizon')

        self.backlog_width = self.backlog_size // self.time_horizon

        self.action_space = spaces.discrete.Discrete(MAXIMUM_QUEUE_SIZE)

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
                self.time_horizon, self.scheduler.total_memory * self.job_slots
            )
        )
        self.processor_slots_space = spaces.box.Box(
            low=0.0, high=1.0, shape=(
                self.time_horizon,
                self.scheduler.number_of_processors * self.job_slots
            )
        )

        self.observation_space = spaces.tuple.Tuple((
            self.processor_space, self.memory_space,
            self.processor_slots_space, self.memory_slots_space,
            self.backlog_space
        ))

        return self.state

    def render(self, mode='human'):
        pass

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
            workload.BinomialWorkloadGenerator(
                self.new_job_rate, self.small_job_chance,
                cpu_dominant_parameters.small, cpu_dominant_parameters.large
            ),
            workload.BinomialWorkloadGenerator(
                self.new_job_rate, self.small_job_chance,
                mem_dominant_parameters.small, mem_dominant_parameters.large
            ),
        )

    def _configure_environment(self) -> DeepRmEnv:
        """Provides a chance for subclasses to specialize behavior."""
        self.memory = AMOUNT_OF_MEMORY
        self.processors = NUMBER_OF_PROCESSORS

        self.n_resources = NUMBER_OF_RESOURCES  # resources in the system
        self.n_work = MAXIMUM_QUEUE_SIZE  # max amount of work in the queue
        self.time_horizon = TIME_HORIZON  # number of time steps in the graph

        self.max_job_len = MAXIMUM_JOB_LENGTH  # max duration of new jobs
        self.max_job_size = MAXIMUM_JOB_SIZE

        self.backlog_size = BACKLOG_SIZE  # backlog queue size
        self.job_num_cap = MAXIMUM_NUMBER_OF_ACTIVE_JOBS

        self.new_job_rate = NEW_JOB_RATE  # rate of job arrival
        self.small_job_chance = SMALL_JOB_CHANCE  # chance a new job is small

        self.job_slots = JOB_SLOTS  # Number of jobs to show

        self.reset()

        return self