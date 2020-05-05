#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum

from abc import ABC, abstractmethod

from . import workload, scheduler as sched


class SimulationType(enum.Enum):
    TIME_BASED = 0
    EVENT_BASED = 1


class Simulator(ABC):
    current_time: int
    scheduler: sched.Scheduler
    simulation_start_time: int

    def __init__(self,
                 workload_generator: workload.WorkloadGenerator,
                 scheduler: sched.Scheduler):

        self.current_time = 0
        self.scheduler = scheduler
        self.simulation_start_time = 0
        self.workload: workload.WorkloadGenerator = workload_generator

    @staticmethod
    def make(simulation_type: SimulationType,
             workload_generator: workload.WorkloadGenerator,
             scheduler: sched.Scheduler):
        if simulation_type == SimulationType.TIME_BASED:
            return TimeBasedSimulator(workload_generator, scheduler)
        else:
            raise RuntimeError(f"Unsupported simulation type {simulation_type}")

    @abstractmethod
    def step(self) -> None:
        "Runs a simulation step."


class TimeBasedSimulator(Simulator):
    scheduler: sched.Scheduler

    def __init__(self, workload_generator: workload.WorkloadGenerator,
                 scheduler: sched.Scheduler):
        super().__init__(workload_generator, scheduler)
        self.current_time = 0

    def step(self, submit=True):
        self.current_time += 1
        self.scheduler.step()
        job = self.workload.sample(self.current_time) if submit else None
        if job:
            self.scheduler.submit(job)

