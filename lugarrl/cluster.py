#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Iterable

from . import resource_pool
from .job import Job

RESOURCE_TYPE = Tuple[Iterable[resource_pool.Interval], Iterable[resource_pool.Interval]]


class Cluster(object):
    def __init__(self, processors, memory, ignore_memory=False):
        self.ignore_memory = ignore_memory
        self.memory = resource_pool.ResourcePool(resource_pool.ResourceType.MEMORY, memory)
        self.processors = resource_pool.ResourcePool(resource_pool.ResourceType.CPU, processors)

    @property
    def free_resources(self) -> Tuple[int, int]:
        return self.processors.free_resources, self.memory.free_resources

    def fits(self, job) -> bool:
        return self.processors.fits(job.requested_processors) and \
               (self.ignore_memory or self.memory.fits(job.requested_memory))

    def allocate(self, job: Job) -> None:
        if not self.fits(job):
            raise AssertionError("Unable to allocate resources for job bigger than cluster")
        job.resources_used.processors = self.processors.find(job.requested_processors)
        job.resources_used.memory = self.memory.find(job.requested_memory)
        self.processors.allocate(job.resources_used.processors)
        self.memory.allocate(job.resources_used.memory)

    def free(self, job: Job) -> None:
        self.processors.deallocate(job.resources_used.processors)
        self.memory.deallocate(job.resources_used.memory)
