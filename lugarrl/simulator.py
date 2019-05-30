#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

from . import workload, scheduler as sched


class Simulator(ABC):
    def __init__(self,
                 workload_generator: workload.WorkloadGenerator,
                 scheduler: sched.Scheduler):
        self.workload = workload_generator

        self.current_time = 0
        self.scheduler = scheduler
        self.simulation_start_time = 0

    @abstractmethod
    def step(self):
        pass


class TimeBasedSimulator(Simulator):
    def __init__(self, workload_generator: workload.WorkloadGenerator,
                 scheduler: sched.Scheduler):
        super().__init__(workload_generator, scheduler)
        self.current_time = 0

    def step(self):
        self.current_time += 1
        self.scheduler.step()
        job = self.workload.sample(self.current_time)
        if job:
            self.scheduler.submit(job)

