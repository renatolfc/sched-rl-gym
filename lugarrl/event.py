#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum
import warnings

from .heap import Heap
from .resource_pool import ResourceType, IntervalTree

class EventType(enum.IntEnum):
    ALLOCATION = 0
    DEALLOCATION = 1


class ResourceEvent:
    def __init__(self, time, event_type: EventType, resource_type: ResourceType, resources: IntervalTree):
        self.time = time
        self.resources = resources
        self.event_type = event_type
        self.resource_type = resource_type


class ResourceEventQueue:
    def __init__(self, time=0):
        self.past = []
        self.time = time
        self.future = Heap()

    def add(self, event: ResourceEvent):
        if event.time >= self.time:
            self.future.add(event, event.time)
        else:
            self.past.append(event)
            self.past.sort(key=lambda e: e.time)
            warnings.warn('Adding events to the past might change the ordering of '
                          'events that happened at the same time.')

    def step(self, time=1):
        if time < 0:
            raise AssertionError("Tried to move into the past.")
        self.time += time
        present = []
        first = self.future.first
        while first and first.time <= self.time:
            current = self.future.pop()
            present.append(current)
            self.past.append(current)
            first = self.future.first
        return present

    @property
    def next(self):
        if len(self.future) == 0:
            return None
        return self.future.first

    @property
    def last(self):
        return self.past[-1]
