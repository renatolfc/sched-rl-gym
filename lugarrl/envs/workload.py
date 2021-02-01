#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import itertools
from math import log2
from typing import Optional
from collections import namedtuple
from parallelworkloads.lublin99 import Lublin99

from lugarrl import workload as wl, job

JobParameters = namedtuple('JobParameters', ['small', 'large'])


class DeepRmWorkloadGenerator(wl.DistributionalWorkloadGenerator):
    def __init__(self, *args: wl.BinomialWorkloadGenerator):
        super().__init__(max([w.length for w in args]))

        self.generators = args
        self.counter = itertools.count(1)

        for generator in self.generators:
            generator.counter = self.counter

    def step(self, offset=1) -> Optional[job.Job]:
        return self.generators[
            random.randint(0, len(self.generators) - 1)
        ].step()

    def __len__(self):
        return self.generators[0].length

    def peek(self):
        return self.step()

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


class SyntheticWorkloadGenerator(wl.TraceGenerator):
    def __init__(self, length, nodes, start_time=8, random_seed=0, restart=False):
        """Synthetic workload generator based on Lublin's work.

        Parameters
        ----------
            length : number of jobs to generate
            nodes : number of compute nodes in the system
            start_time : hour of day in which to start simulation
            random_seed : random seed to use to generate jobs
        """
        self.lublin = Lublin99(True, random_seed)#, length)
        self.lublin.start = start_time

        uLow = .8
        uProb = .7
        uHi = log2(nodes)
        uMed = ((uHi - 1.5) + (uHi - 3.5)) / 2

        self.lublin.setParallelJobProbabilities(
            False, uLow, uMed, uHi, uProb
        )
        self.lublin.setParallelJobProbabilities(
            True, uLow, uMed, uHi, uProb
        )
        trace = self.refresh_jobs()
        super().__init__(restart, trace)

    def refresh_jobs(self):
        self.trace = [job.Job.from_swf_job(j) for j in self.lublin.generate()]
        return self.trace


def build(workload_config: dict):
    type = workload_config['type']
    kwargs = {
        k: v for k, v in workload_config.items() if k != 'type'
    }
    if type == 'deeprm':
        return DeepRmWorkloadGenerator.build(**kwargs)
    elif type == 'lublin':
        return SyntheticWorkloadGenerator(**kwargs)
