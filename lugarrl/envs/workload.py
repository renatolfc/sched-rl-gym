#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import itertools
from typing import Optional
from collections import namedtuple

from lugarrl import workload as wl, job

JobParameters = namedtuple('JobParameters', ['small', 'large'])


class DeepRmWorkloadGenerator(wl.DistributionalWorkloadGenerator):
    def __init__(self, *args: wl.BinomialWorkloadGenerator):
        super().__init__(max([w.length for w in args]))

        self.generators = args
        self.counter = itertools.count(1)

        for generator in self.generators:
            generator.counter = self.counter

    def sample(self, submission_time=0) -> Optional[job.Job]:
        return self.generators[
            random.randint(0, len(self.generators) - 1)
        ].sample(submission_time)

    @staticmethod
    def build(new_job_rate, small_job_chance,
              max_job_len, max_job_size):
        # Time-related job parameters {{{
        small_job_time_lower = 1
        small_job_time_upper = max(max_job_len // 5, 1)
        large_job_time_lower = int(max_job_len * (2 / 3))
        large_job_time_upper = max_job_len
        # }}}

        # Resource-related job parameters {{{
        dominant_resource_lower = max_job_size // 2
        dominant_resource_upper = max_job_size
        other_resource_lower = 1
        other_resource_upper = max_job_size // 5
        # }}}

        cpu_dominant_parameters = JobParameters(  # {{{
            job.JobParameters(
                small_job_time_lower,
                small_job_time_upper,
                dominant_resource_lower,
                dominant_resource_upper,
                other_resource_lower,
                other_resource_upper
            ),
            job.JobParameters(
                large_job_time_lower,
                large_job_time_upper,
                dominant_resource_lower,
                dominant_resource_upper,
                other_resource_lower,
                other_resource_upper
            ),
        )  # }}}

        mem_dominant_parameters = JobParameters(  # {{{
            job.JobParameters(
                small_job_time_lower,
                small_job_time_upper,
                other_resource_lower,
                other_resource_upper,
                dominant_resource_lower,
                dominant_resource_upper,
            ),
            job.JobParameters(
                large_job_time_lower,
                large_job_time_upper,
                other_resource_lower,
                other_resource_upper,
                dominant_resource_lower,
                dominant_resource_upper,
            ),
        )  # }}}

        return DeepRmWorkloadGenerator(
            wl.BinomialWorkloadGenerator(
                new_job_rate, small_job_chance,
                cpu_dominant_parameters.small, cpu_dominant_parameters.large
            ),
            wl.BinomialWorkloadGenerator(
                new_job_rate, small_job_chance,
                mem_dominant_parameters.small, mem_dominant_parameters.large
            ),
        )