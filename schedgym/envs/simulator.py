#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import IntEnum
from typing import Union, Optional

from schedgym.scheduler import NullScheduler
from schedgym.envs.workload import (
    DeepRmWorkloadGenerator,
    SyntheticWorkloadGenerator
)


class SimulationType(IntEnum):
    EVENT_BASED = 0,
    TIME_BASED = 1

    @staticmethod
    def from_str(simulation_type: str):
        simulation_type = simulation_type.upper().replace('-', '_')
        if simulation_type in SimulationType.__members__:
            return SimulationType[simulation_type]
        else:
            raise ValueError(
                f'{simulation_type} is not a valid SimulationType.'
            )


class DeepRmSimulator:
    scheduler: NullScheduler
    workload: Union[DeepRmWorkloadGenerator, SyntheticWorkloadGenerator]

    def __init__(self, workload_generator: Union[DeepRmWorkloadGenerator,
                                                 SyntheticWorkloadGenerator],
                 scheduler: NullScheduler,
                 simulation_type: SimulationType = SimulationType.TIME_BASED,
                 job_slots: Optional[int] = None):

        self.scheduler = scheduler
        self.workload = workload_generator
        self.simulation_type = simulation_type
        self.job_slots = slice(0, job_slots)
        self.simulator = self.build()
        self.reset(self.workload, scheduler)

    def rl_step(self, action: Optional[int]) -> bool:
        return self.simulator.rl_step(action)

    def build(self):
        if self.simulation_type == SimulationType.EVENT_BASED:
            return EventBasedDeepRmSimulator(
                self.workload,
                self.scheduler,
                self.job_slots,
            )
        elif self.simulation_type == SimulationType.TIME_BASED:
            return TimeBasedDeepRmSimulator(
                self.workload,
                self.scheduler,
                self.job_slots,
            )
        else:
            raise NotImplementedError(
                f"Unsupported simulation type {self.simulation_type}"
            )

    @property
    def current_time(self):
        return self.simulator.current_time

    @property
    def last_job_time(self):
        return self.simulator.last_job_time

    def reset(self, workload, scheduler):
        self.scheduler = scheduler
        self.workload = workload
        self.simulator = self.build()


class EventBasedDeepRmSimulator:
    last_job_time: int
    scheduler: NullScheduler
    job_slots: slice

    def __init__(self, workload_generator: DeepRmWorkloadGenerator,
                 scheduler: NullScheduler, job_slots: slice):
        if (not isinstance(workload_generator, DeepRmWorkloadGenerator)
                and not isinstance(workload_generator,
                                   SyntheticWorkloadGenerator)) \
                or not isinstance(scheduler, NullScheduler):
            raise AssertionError("Invalid arguments received.")

        self.current_time = 0
        self.scheduler = scheduler
        self.simulation_start_time = 0
        self.workload = workload_generator
        self.job_slots = job_slots

        self.current_time = self.last_job_time = 0
        if isinstance(workload_generator, SyntheticWorkloadGenerator):
            first_job_time = workload_generator.peek().submission_time - 1
            workload_generator.current_time = first_job_time
            scheduler.job_events.time = first_job_time
            scheduler.current_time = first_job_time
            self.current_time = first_job_time

    def rl_step(self, action: int) -> bool:
        "Returns True when an action executes successfully."

        if self.scheduler.step(action):
            return False

        self.current_time += 1
        while True:
            j = self.workload.step()
            if j:
                self.scheduler.submit(j)
                self.last_job_time = self.current_time
            self.scheduler.forward_time()
            if self.scheduler.some_job_fits(self.job_slots):
                break
        return True


class TimeBasedDeepRmSimulator:
    last_job_time: int
    scheduler: NullScheduler
    job_slots: slice

    def __init__(self, workload_generator: DeepRmWorkloadGenerator,
                 scheduler: NullScheduler, job_slots: slice):
        if (not isinstance(workload_generator, DeepRmWorkloadGenerator)
                and not isinstance(
                    workload_generator, SyntheticWorkloadGenerator
                )) \
                or not isinstance(scheduler, NullScheduler):
            raise AssertionError("Invalid arguments received.")

        self.scheduler = scheduler
        self.simulation_start_time = 0
        self.workload = workload_generator
        self.current_time = self.last_job_time = 0
        self.job_slots = job_slots

        if isinstance(workload_generator, SyntheticWorkloadGenerator):
            first_job_time = workload_generator.peek().submission_time - 1
            workload_generator.current_time = first_job_time
            scheduler.job_events.time = first_job_time
            scheduler.current_time = first_job_time

    def step(self, submit=True):
        "Not implemented in DeepRmSimulator"
        raise NotImplementedError('This simulator cannot follow the base API')

    def rl_step(self, action: int) -> bool:
        "Returns True when time passes."

        if self.scheduler.step(action):
            return False
        else:
            self.current_time += 1
            j = self.workload.step()
            if j:
                self.scheduler.submit(j)
                self.last_job_time = self.current_time
            self.scheduler.forward_time()
            return True