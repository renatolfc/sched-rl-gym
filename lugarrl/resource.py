#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""resource - basic resource unit

This module has two classes:
  1. `PrimaryResource`, an enumeration for the different supported types (CPU
     and MEMORY)
  2. The basic resource group, which is comprised of *both* CPU and memory
"""

import copy
import enum
from typing import Tuple

from intervaltree import IntervalTree


class PrimaryResource(enum.IntEnum):
    """Enumeration for identifying the various supported resource types."""
    CPU = 0
    MEMORY = 1


class Resource(object):
    """The basic resource group.

    This groups IntervalTrees into as many resources that can are supported in
    the system.

    This is referenced by a :class:`lugarrl.job.Job` to represent *which
    specific resources* are being used by that job.

    Parameters
    ----------
        processors : IntervalTree
            An interval tree that defines a set of processors
        memory : IntervalTree
            An interval tree that defines a set of memory resources
        ignore_memory : bool
            Whether memory should be taken in consideration when measuring
            resource usage.
    """
    memory: IntervalTree
    """IntervalTree that stores memory used"""
    processors: IntervalTree
    """IntervalTree that stores processors used"""

    def __init__(self, processors: IntervalTree = IntervalTree(),
                 memory: IntervalTree = IntervalTree(),
                 ignore_memory: bool = False):
        self.ignore_memory = ignore_memory
        self.processors = copy.copy(processors)
        self.memory = copy.copy(memory)

    def measure(self) -> Tuple[int, int]:
        """Returns the total amount of resources in use.

        Returns:
            Tuple: A tuple containing the amount of resources used for each
            resource type supported.
        """
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
