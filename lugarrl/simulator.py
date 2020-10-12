#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""simulator - Classes for simulating job submission and execution.

This module comprises an abstract base class for simulation and a time-based
simulator that inherits directly from `Simulator`.

The time-based simulator is coupled with
a :class:`lugarrl.workload.WorkloadGenerator` to generate jobs at a given time
step.
"""

import enum

from abc import ABC, abstractmethod

from . import workload, scheduler as sched


class SimulationType(enum.Enum):
    "Enumeration to differentiate between simulation types"
    TIME_BASED = 0
    EVENT_BASED = 1


class Simulator(ABC):
    """Abstract base class for simulation.

    Parameters
    ----------
        workload_generator : workload.WorkloadGenerator
            An object to generate load when time is stepped.
        scheduler : sched.Scheduler
            A scheduling algorithm that will schedule jobs according to a given
            rule.
    """
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
        """Factory method for instantiating new simulators."""
        if simulation_type == SimulationType.TIME_BASED:
            return TimeBasedSimulator(workload_generator, scheduler)
        raise RuntimeError(f"Unsupported simulation type {simulation_type}")

    @abstractmethod
    def step(self, submit) -> None:
        "Runs a simulation step."


class TimeBasedSimulator(Simulator):
    """A simulator that is based on time."""
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
