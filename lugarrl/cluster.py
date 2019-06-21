#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Tuple, Iterable

import numpy as np

from . import pool

from .job import Job, Resource
from .event import JobEvent, EventType

RESOURCE_TYPE = Tuple[Iterable[pool.Interval], Iterable[pool.Interval]]


class Cluster(object):
    def __init__(self, processors, memory, ignore_memory=False, used_processors=None, used_memory=None):
        self.ignore_memory = ignore_memory
        self.memory = pool.ResourcePool(pool.ResourceType.MEMORY, memory, used_memory)
        self.processors = pool.ResourcePool(pool.ResourceType.CPU, processors, used_processors)

    @property
    def free_resources(self) -> Tuple[int, int]:
        return self.processors.free_resources, self.memory.free_resources

    def fits(self, job) -> bool:
        return self.processors.fits(job.requested_processors) and \
               (self.ignore_memory or self.memory.fits(job.requested_memory))

    def allocate(self, job: Job) -> None:
        if not self.fits(job):
            raise AssertionError(f"Unable to allocate resources for {job} in {self}")
        self.processors.allocate(job.resources_used.processors)
        self.memory.allocate(job.resources_used.memory)

    def clone(self):
        return copy.deepcopy(self)

    def find(self, job: Job) -> Resource:
        p = self.processors.find(job.requested_processors, job.id)
        if not p:
            return Resource()
        if self.ignore_memory:
            return Resource(p, ignore_memory=True)
        m = self.memory.find(job.requested_memory, job.id)
        return Resource(p, m)

    def free(self, job: Job) -> None:
        self.processors.free(job.resources_used.processors)
        if not self.ignore_memory:
            self.memory.free(job.resources_used.memory)

    def find_resources_at_time(self, time: int, job: Job, events: Iterable[JobEvent]) -> Resource:
        used = Resource(self.processors.used_pool, self.memory.used_pool)
        for event in (e for e in events if (time + 1 <= e.time < job.requested_time + time and
                                            e.event_type == EventType.JOB_START)):
            for i in event.processors:
                used.processors.add(i)
            for i in event.memory:
                used.memory.add(i)
        used.processors.merge_overlaps()
        used.memory.merge_overlaps()
        return Cluster(
            self.processors.size, self.memory.size, self.ignore_memory, used.processors, used.memory
        ).find(job)

    @property
    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        processors = np.zeros(self.processors.size)
        for i in self.processors.used_pool:
            processors[i.begin:i.end] = i.data
        memory = np.zeros(self.memory.size)
        for i in self.memory.used_pool:
            memory[i.begin:i.end] = i.data
        return processors, memory

    def get_job_state(self, job: Job, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
        processors = np.zeros((timesteps, self.processors.size))
        memory = np.zeros((timesteps, self.memory.size))
        processors[:job.requested_time, job.requested_processors] = 1.0
        memory[:job.requested_time, job.requested_processors] = 1.0
        return processors, memory

    def __bool__(self):
        return self.processors.free_resources != 0 and self.memory.free_resources != 0

    def __repr__(self):
        return f'Cluster({self.processors}, {self.memory}, {self.ignore_memory})'

    def __str__(self):
        return f'Cluster({self.processors}, {self.memory}, {self.ignore_memory})'
