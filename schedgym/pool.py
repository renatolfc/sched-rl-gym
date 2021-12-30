#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pool - Resource Pool management (see :class:`schedgym.cluster.Cluster`)."""

import copy
import enum
from typing import Iterable, List, Optional

from intervaltree import IntervalTree, Interval


class ResourceType(enum.IntEnum):
    """Enumeration to determine which kind of resource we're managing."""

    CPU = 1
    MEMORY = 0


class ResourcePool:
    """A pool of resources.

    This is the basic structure managed by a :class:`schedgym.cluster.Cluster`.

    Parameters
    ----------
        resource_type : ResourceType
            The type of resource in this pool
        size : int
            The amount of resources available in this pool
        used_pool : IntervalTree
            The set of resources currently in use in this resource pool.
    """

    used_pool: IntervalTree

    def __init__(
        self,
        resource_type: ResourceType,
        size: int,
        used_pool: IntervalTree = None,
    ):
        self.size = size
        self.used_resources = 0
        self.type = resource_type
        if used_pool is None:
            self.used_pool = IntervalTree()
        else:
            self.used_pool = used_pool
            self.used_resources = sum(
                [ResourcePool.measure(i) for i in used_pool]
            )

    def clone(self):
        """Duplicates this ResourcePool in memory."""
        return copy.deepcopy(self)

    @property
    def free_resources(self) -> int:
        """Returns the amount of free resources in this resource pool"""
        return self.size - self.used_resources

    def fits(self, size) -> bool:
        """Checks whether a given amount of resources can be allocated.

        Parameters
        ----------
            size : int
                The amount of resources to allocate in this pool

        Returns:
            bool: True when the size fits the pool, and False otherwise.
        """
        if size <= 0:
            raise AssertionError("Can't allocate zero resources")
        return size <= self.free_resources

    @staticmethod
    def measure(interval: Interval):
        """Measures the size of an interval.

        Parameters
        ----------
            interval : Interval
                The interval to be measured.
        """
        return interval.end - interval.begin

    def find(self, size: int, data: Optional[int] = None) -> IntervalTree:
        """Finds an interval tree of a given size in this resource pool.

        This is essentially an operation to find *which* resources to allocate
        considering that we manage individual resource units and guarantee
        exclusive usage by a resource unit.

        Parameters
        ----------
            size : int
                The size (amount) of resources to allocate
            data : Optional[int]
                The identifier of the "owner" of the found resources. This
                allows us to keep track which job "owns" which resources during
                execution.

        Returns:
            IntervalTree: An interval tree with the size requested if such
            a tree can be found. Otherwise, an empty tree is returned.
        """
        used = IntervalTree()
        if not self.fits(size):
            return used
        free = IntervalTree([Interval(0, self.size, data)])
        used_size: int = 0
        for interval in self.used_pool:
            free.chop(interval.begin, interval.end)
        for interval in free:
            temp_size = ResourcePool.measure(interval) + used_size
            if temp_size == size:
                used.add(interval)
                break
            if temp_size < size:
                used.add(interval)
                used_size = temp_size
            else:
                used.add(
                    Interval(
                        interval.begin, interval.begin + size - used_size, data
                    )
                )
                break
        return used

    def allocate(self, intervals: Iterable[Interval]) -> None:
        """Adds a set of intervals to the current used pool of resources.

        This is the opposite of :func:`schedgym.cluster.Cluster.free`.

        Parameters
        ----------
            intervals : Iterable[Interval]
                The set of intervals that should be allocated (most likely,
                this will be the resource of calling
                :func:`schedgym.cluster.Cluster.find`).

        Returns:
            None
        """
        for i in intervals:
            if self.used_resources + self.measure(i) > self.size:
                raise AssertionError(
                    'Tried to allocate past size of resource pool'
                )
            self.used_pool.add(i)
            self.used_resources += self.measure(i)

    def free(self, intervals: Iterable[Interval]) -> None:
        """Frees a set of used resources.

        This is the opposite of :func:`schedgym.cluster.Cluster.allocate`.

        Parameters
        ----------
            intervals : Iterable[Interval]
                The set of intervals to be freed (most likely, these will have
                been allocated with the output of
                :func:`schedgym.cluster.Cluster.find`).
        """
        for i in intervals:
            if i not in self.used_pool:
                raise AssertionError('Tried to free unused resource set')
            self.used_pool.remove(i)
            self.used_resources -= self.measure(i)

    @property
    def intervals(self) -> List[Interval]:
        """The set of intervals currently used in this resource pool."""
        # pylint: disable=unnecessary-comprehension
        return [i for i in self.used_pool]

    def __repr__(self):
        return (
            f'ResourcePool(resource_type={self.type}, '
            f'size={self.size}, used_pool={self.used_pool})'
        )
