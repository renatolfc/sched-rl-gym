#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""cluster - Classes for cluster management

The workhorse of this module is the :class:`lugarrl.cluster.Cluster` class,
which manages resources in a cluster.
"""

import copy
from typing import Tuple, Iterable, Optional

import numpy as np

from . import pool

from .job import Job, Resource
from .event import JobEvent, EventType

# pylint: disable=C
RESOURCE_TYPE = Tuple[Iterable[pool.Interval], Iterable[pool.Interval]]


class Cluster:
    """A cluster as a set of resources.

    Currently, this doesn't make a distinction between machines. So it only
    manages groups of resources.

    Note that although we don't differentiate between machines, we **do** honor
    resources. Therefore, if a given processor is allocated by a job j, we make
    sure not to allocate processor n for any other job until j finishes.

    Due to the above constraint, some checks are more complex (and,
    consequently, slower) than if we disregarded *which* processors and memory
    units were used and only counted the *amount* of resources used.

    This makes our design slightly closer to reality, though.

    The figure below shows the relationship between clusters, ResourcePools,
    and the basic data structure for resource management (`IntervalTree`).

    .. image:: /img/cluster-resourcepool.svg

    Parameters
    ----------
        processors : int
            The number of processors in this cluster
        memory : int
            The amount of memory in this cluster
        ignore_memory : bool
            Whether memory should be considered for decisions or not
        used_processors : Optional[Resource]
            Processors already in use in this cluster
        used_memory : Optional[Resource]
            Amount of memory already used in this cluster
    """
    ignore_memory: bool
    memory: pool.ResourcePool
    processors: pool.ResourcePool

    def __init__(self, processors: int, memory: int,
                 ignore_memory: bool = False,
                 used_processors: Optional[Resource] = None,
                 used_memory: Optional[Resource] = None):
        self.ignore_memory = ignore_memory
        self.memory = pool.ResourcePool(
            pool.ResourceType.MEMORY, memory, used_memory
        )
        self.processors = pool.ResourcePool(
            pool.ResourceType.CPU, processors, used_processors
        )

    @property
    def free_resources(self) -> Tuple[int, int]:
        """The set of resources *not* in use in this cluster."""
        return self.processors.free_resources, self.memory.free_resources

    def fits(self, job: Job) -> bool:
        """Checks whether a job fits in this cluster.

        Parameters
        ----------
            job : Job
                The job to check against in this cluster

        Returns:
            True if the job fits the cluster (can be added to the cluster), and
            False otherwise
        """
        return self.processors.fits(job.requested_processors) and \
            (self.ignore_memory or self.memory.fits(job.requested_memory))

    def allocate(self, job: Job) -> None:
        """Checks whether a job fits the system and allocates resources for it.

        Parameters
        ----------
            job : Job
                The job to allocate resources to.
        """
        if not self.fits(job):
            raise AssertionError(
                f"Unable to allocate resources for {job} in {self}"
            )
        self.processors.allocate(job.resources.processors)
        self.memory.allocate(job.resources.memory)

    def clone(self):
        """Clones this Cluster (duplicating it in memory)."""
        return copy.deepcopy(self)

    def find(self, job: Job) -> Resource:
        """Finds resources for a job.

        If the job fits in the system, this will return a set of resources that
        can be used by a job. If it doesn't, will return an empty set of
        resources (which evaluate to False in boolean expressions).

        Parameters
        ----------
            job : Job
                The job to find resources to.
        """
        p = self.processors.find(job.requested_processors, job.id)
        if not p:
            return Resource()
        if self.ignore_memory:
            return Resource(p, ignore_memory=True)
        m = self.memory.find(job.requested_memory, job.id)
        return Resource(p, m)

    def free(self, job: Job) -> None:
        """Frees the resources used by a job.

        Parameters
        ----------
            job : Job
                The job to free resources from.
        """
        self.processors.free(job.resources.processors)
        if not self.ignore_memory:
            self.memory.free(job.resources.memory)

    def find_resources_at_time(self, time: int, job: Job, events:
                               Iterable[JobEvent]) -> Resource:
        """Finds resources for a job at a given time step.

        This is probably the most complex (and most important) function in this
        class. To find an allocation for a job, we have to iterate through the
        queue of events and evaluating the state of the system given that set
        of events to check whether a given job would fit the system.

        Since this method can be called with time stamps in the far future, we
        are required to play events to find the exact configuration in the
        future.

        Parameters
        ----------
            time : int
                The time at which to check whether the job fits the system
            job : Job
                The job to check
            events : Iterable[JobEvent]
                A set of events that will play out in the future

        Returns:
            A set of resources if the job fits the cluster at time `time`, or
            an empty set of resources otherwise. (See
            :func:`lugarrl.cluster.Cluster.find`.)
        """
        used = Resource(self.processors.used_pool, self.memory.used_pool)
        valid = lambda e, time: time + 1 <= e.time < job.requested_time + time
        for event in (e for e in events if (valid(e, time) and
                                            e.type == EventType.JOB_START)):
            for i in event.processors:
                used.processors.add(i)
            for i in event.memory:
                used.memory.add(i)
        used.processors.merge_overlaps()
        used.memory.merge_overlaps()
        return Cluster(
            self.processors.size, self.memory.size, self.ignore_memory,
            used.processors, used.memory
        ).find(job)

    @property
    def state(self) -> Tuple[np.ndarray, ...]:
        """Gets the current state of the cluster as numpy arrays.

        Returns:
            Tuple: a pair containing the number of processors used and the
            memory used and the jobs that are using such resources.
        """
        processors = (
            self.processors.free_resources,
            self.processors.used_resources,
            {(i.begin, i.end): i.data for i in self.processors.used_pool}
        )
        memory = (
            self.memory.free_resources,
            self.memory.used_resources,
            {(i.begin, i.end): i.data for i in self.memory.used_pool}
        )
        if self.ignore_memory:
            return (processors,)
        else:
            return processors, memory

    def get_job_state(self, job: Job, timesteps: int) -> Tuple[np.ndarray, ...]:
        """Gets the stat of a job given a time horizon.

        Parameters
        ----------
            timesteps : int
                The number of time steps to look into the future to create this
                representation.
        """
        processors = np.zeros((timesteps, self.processors.size))
        memory = np.zeros((timesteps, self.memory.size))
        processors[:job.requested_time, :job.requested_processors] = 1.0
        if self.ignore_memory:
            return (processors,)
        memory[:job.requested_time, :job.requested_memory] = 1.0
        return processors, memory

    def __bool__(self):
        return self.processors.free_resources != 0 and\
            self.memory.free_resources != 0

    def __repr__(self):
        return \
            f'Cluster({self.processors}, {self.memory}, {self.ignore_memory})'

    def __str__(self):
        return \
            f'Cluster({self.processors}, {self.memory}, {self.ignore_memory})'
