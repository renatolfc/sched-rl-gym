#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""distribution - Generative models for workload generation"""

import math
import random
import itertools
from abc import ABC, abstractmethod
from typing import List, Optional

from schedgym.job import Job, JobParameters
from schedgym.workload.base import WorkloadGenerator


class DistributionalWorkloadGenerator(WorkloadGenerator, ABC):
    """An abstract class for workload generation based on distributions.

    Parameters
    ----------
        length : int
            An optional length of workload generation. When length samples
            are generated, automatic iteration will stop.
    """
    length: int
    current_element: int

    def __init__(self, length=0):
        self.length = length
        self.current_element = 0

    @abstractmethod
    def step(self, offset=1) -> List[Optional[Job]]:
        """Steps the workload generator by :param offset:.

        This may, or may not, return new jobs, depending on the internal
        probability distributions of the workload generator.

        Parameters
        ----------
            offset : int
                The number of time steps to advance the workload generator.
         """


class BinomialWorkloadGenerator(DistributionalWorkloadGenerator):
    """A workload generator that is based on a Bernoulli distribution.

    Parameters
    ----------
        new_job_rate : float
            The probability of generating a new job
        small_job_chance : float
            The probability a sampled job will be "small"
        small_job_parameters : JobParameters
            The characteristics of "small" jobs
        large_job_parameters : JobParameters
            The characteristics of "large" jobs
        length : int
            The size of the sequence of jobs generated when iterating over this
            workload generator
    """
    new_job_rate: float
    small_job_chance: float
    large_job: JobParameters
    small_job: JobParameters

    def __init__(self, new_job_rate, small_job_chance, small_job_parameters,
                 large_job_parameters, length=0, runtime_estimates=None,
                 estimate_parameters=None):
        super().__init__(length)

        self.current_time = 0
        self.counter = itertools.count(1)
        self.new_job_rate = new_job_rate
        self.small_job_chance = small_job_chance
        self.small_job = small_job_parameters
        self.large_job = large_job_parameters

        if runtime_estimates is not None and \
                runtime_estimates not in [
                    'gaussian', 'gaussian-over', 'gaussian-under']:
            raise ValueError(f'Unsupported estimate type {runtime_estimates}')

        self.runtime_estimates = runtime_estimates
        self.estimate_parameters = estimate_parameters

    def step(self, offset=1) -> List[Optional[Job]]:
        self.current_time += offset
        if random.random() > self.new_job_rate:
            return []
        if random.random() < self.small_job_chance:
            j = self.small_job.sample(self.current_time)
        else:
            j = self.large_job.sample(self.current_time)
        if self.runtime_estimates and \
                self.runtime_estimates.startswith('gaussian'):
            diff = random.gauss(
                0,
                self.estimate_parameters * j.execution_time
            )
            if 'over' in self.runtime_estimates:
                diff = abs(diff)
            elif 'under' in self.runtime_estimates:
                diff = -abs(diff)
            j.requested_time = max(math.ceil(j.execution_time + diff), 1)
        j.id = next(self.counter)
        return [j]

    def __len__(self):
        return self.length

    def peek(self):
        return self.step(0)
