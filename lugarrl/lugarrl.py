#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import simulator, workload, job, fifo_scheduler


class LugarRL(object):
    def __init__(self):
        self.small_job_parameters = job.JobParameters(1, 3, 1, 2, 2, 16)
        self.large_job_parameters = job.JobParameters(10, 15, 4, 8, 32, 64)
        self.workload = workload.BinomialWorkloadGenerator(
            0.7, 0.8, self.small_job_parameters, self.large_job_parameters
        )
        self.scheduler = fifo_scheduler.FifoScheduler(32, 1024)
        self.simulator = simulator.TimeBasedSimulator(
            self.workload, self.scheduler
        )

    def step(self) -> None:
        self.simulator.step()
