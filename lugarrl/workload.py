#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

from lugarrl.job import JobParameters


class WorkloadGenerator(ABC):
    @abstractmethod
    def __next__(self):
        "Next element in iterator."

    @abstractmethod
    def __iter__(self):
        "Iterator."

    @abstractmethod
    def sample(self, submission_time=0):
        "Sample a job with submission time equal to :param submission_time:."


class DistributionalWorkloadGenerator(WorkloadGenerator, ABC):
    length: int
    current_element: int

    def __init__(self, length=0):
        self.length = length
        self.current_element = 0

    def __next__(self):
        if self.length and self.current_element >= self.length:
            raise StopIteration()
        self.current_element += 1
        return self.sample()

    def __iter__(self):
        return self


class BinomialWorkloadGenerator(DistributionalWorkloadGenerator):
    small_job_chance: float
    small_job: JobParameters
    large_job: JobParameters

    def __init__(self, new_job_rate, small_job_chance, small_job_parameters, large_job_parameters, length=0):
        super().__init__(length)

        self.new_job_rate = new_job_rate
        self.small_job_chance = small_job_chance
        self.small_job = small_job_parameters
        self.large_job = large_job_parameters

    def sample(self, submission_time=0):
        if np.random.rand() > self.new_job_rate:
            return None
        else:
            if np.random.rand() < self.small_job_chance:
                return self.small_job.sample(submission_time)
            else:
                return self.large_job.sample(submission_time)
