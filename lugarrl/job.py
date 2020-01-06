#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import enum

import random
import warnings
from typing import Iterable, NamedTuple, Tuple

import numpy as np

from .pool import Interval, IntervalTree, ResourcePool


class PrimaryResource(enum.IntEnum):
    CPU = 0
    MEMORY = 1


class JobStatus(enum.IntEnum):
    SUBMITTED = 0
    RUNNING = 1
    WAITING = 2
    COMPLETED = 3
    SCHEDULED = 4


class SwfJobStatus(enum.IntEnum):
    FAILED = 0
    COMPLETED = 1
    PARTIAL_TO_BE_CONTINUED = 2
    PARTIAL_LAST_COMPLETED = 3
    PARTIAL_LAST_FAILED = 4
    CANCELLED = 5
    MEANINGLESS = -1


class Resource(object):
    processors: IntervalTree = IntervalTree()
    memory: IntervalTree = IntervalTree()

    def __init__(self, processors: IntervalTree = IntervalTree(),
                 memory: IntervalTree = IntervalTree(),
                 ignore_memory: bool = False):
        self.ignore_memory = ignore_memory
        self.processors = copy.copy(processors)
        self.memory = copy.copy(memory)

    def measure(self) -> Tuple[int, int]:
        processors = sum([i.end - i.begin for i in self.processors])
        memory = sum([i.end - i.begin for i in self.memory])
        return processors, memory

    def __bool__(self) -> bool:
        return bool(self.processors) and \
            (self.ignore_memory or bool(self.memory))

    def __repr__(self):
        return f'Resource({self.processors}, {self.memory})'

    def __str__(self):
        return f'Resource({self.processors}, {self.memory})'


class Job(object):
    resources: Resource

    def __init__(self, job_id, submission_time, execution_time, processors_allocated, average_cpu_use, memory_use,
                 requested_processors, requested_time, requested_memory, status, user_id, group_id, executable,
                 queue_number, partition_number, preceding_job_id, think_time, wait_time, ignore_memory=True):
        self.id = job_id
        self.submission_time = submission_time
        self.execution_time = execution_time
        self.requested_time = requested_time
        self.requested_processors = requested_processors
        self.processors_allocated = processors_allocated
        self.average_cpu_use = average_cpu_use
        self.memory_use = memory_use
        self.requested_memory = requested_memory
        self.status = status
        self.user_id = user_id
        self.group_id = group_id
        self.executable = executable
        self.queue_number = queue_number
        self.partition_number = partition_number
        self.preceding_job_id = preceding_job_id
        self.think_time = think_time
        self.wait_time = wait_time

        self.resources = Resource()
        self.first_scheduling_promise = None
        self.start_time = None
        self.finish_time = None
        self.ignore_memory = ignore_memory
        self.slot_position = None

    def __str__(self):
        return f'Job<{self.id}, {self.status.name}, start={self.start_time}, processors={self.requested_processors}, ' \
            f'memory={self.requested_memory}>'

    __repr__ = __str__

    @property
    def proper(self):
        processors, memory = self.resources.measure()
        return processors == self.requested_processors and (self.ignore_memory or memory == self.requested_memory)

    def slowdown(self):
        try:
            return (self.finish_time - self.submission_time) / self.execution_time
        except TypeError:
            warnings.warn(f"Failed to obtain slowdown for job {self}. It may not have finished yet.")
            return -1


class JobParameters(object):
    lower_time_bound: int
    upper_time_bound: int
    lower_resource_bound: int
    upper_resource_bound: int

    @staticmethod
    def _validate_parameters(*args):
        for param in args:
            if param <= 0:
                raise AssertionError("Unable to work with non-positive bounds.")

    def __init__(self, lower_time_bound: int, upper_time_bound: int, lower_cpu_bound: int,
                 upper_cpu_bound: int, lower_mem_bound: int, upper_mem_bound: int):
        self._validate_parameters(lower_time_bound, upper_time_bound, lower_cpu_bound, upper_cpu_bound,
                                  lower_mem_bound, upper_mem_bound)

        self.lower_time_bound = lower_time_bound
        self.upper_time_bound = upper_time_bound
        self.lower_cpu_bound = lower_cpu_bound
        self.upper_cpu_bound = upper_cpu_bound
        self.lower_mem_bound = lower_mem_bound
        self.upper_mem_bound = upper_mem_bound

        self.resource_samplers = {
            PrimaryResource.CPU: lambda: random.randint(self.lower_cpu_bound, self.upper_cpu_bound),
            PrimaryResource.MEMORY: lambda: random.randint(self.lower_mem_bound, self.upper_mem_bound),
        }

        self.job_id = 1
        self.time_step = 0

    def add_time(self, steps: int = 1) -> None:
        if steps < 0:
            raise AssertionError("Time can't be negative.")
        self.time_step += steps

    def sample(self, submission_time: int = 0) -> Job:
        time_duration = random.randint(self.lower_time_bound, self.upper_time_bound)

        cpu = self.resource_samplers[PrimaryResource.CPU]()
        mem = self.resource_samplers[PrimaryResource.MEMORY]()

        job = Job(
            self.job_id, submission_time if submission_time else self.time_step, time_duration, cpu, 0, mem, cpu,
            time_duration, mem, JobStatus.WAITING, 1, 1, 1, 1, 1, -1, -1, -1
        )
        self.job_id += 1

        return job
