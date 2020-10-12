#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import enum
from typing import Tuple

from intervaltree import IntervalTree


class PrimaryResource(enum.IntEnum):
    CPU = 0
    MEMORY = 1


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
