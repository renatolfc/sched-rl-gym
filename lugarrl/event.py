#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import enum
import warnings
from typing import List, Optional, Iterable, TypeVar, Generic, Iterator

from job import Job
from .heap import Heap
from .resource_pool import ResourceType, Interval

T = TypeVar('T', bound='Event')


class EventType(enum.IntEnum):
    RESOURCE_ALLOCATE = 0
    RESOURCE_FREE = 1
    JOB_START = 3
    JOB_FINISH = 4


class Event(object):
    time: int
    event_type: EventType

    def __init__(self, time: int, event_type: EventType):
        self.time = time
        self.event_type = event_type

    def clone(self):
        return copy.copy(self)


class ResourceEvent(Event):
    resources: Iterable[Interval]
    resource_type: ResourceType

    def __init__(self, time: int, event_type: EventType, resource_type: ResourceType, resources: Iterable[Interval]):
        super().__init__(time, event_type)
        self.resources = resources
        self.resource_type = resource_type


class JobEvent(Event):
    job: Job

    def __init__(self, time: int, event_type: EventType, job: Job):
        super().__init__(time, event_type)
        self.job = job

    @property
    def processors(self):
        return self.job.requested_processors

    @property
    def memory(self):
        return self.job.requested_memory


class EventQueue(Generic[T]):
    time: int
    future: Heap
    past: List[T]

    def __init__(self, time: int = 0):
        self.past = []
        self.time = time
        self.future = Heap()

    def add(self, event: T):
        if event.time >= self.time:
            self.future.add(event, event.time)
        else:
            self.past.append(event)
            self.past.sort(key=lambda e: e.time)
            warnings.warn('Adding events to the past might change the ordering of '
                          'events that happened at the same time.')

    def step(self, time: int = 1) -> Iterable[T]:
        if time < 0:
            raise AssertionError("Tried to move into the past.")
        self.time += time
        present: List[T] = []
        first = self.future.first
        while first and first.time <= self.time:
            current = self.future.pop()
            present.append(current)
            self.past.append(current)
            first = self.future.first
        return present

    @property
    def next(self) -> Optional[T]:
        if len(self.future) == 0:
            return None
        return self.future.first

    @property
    def last(self) -> Optional[T]:
        return self.past[-1] if self.past else None

    def __iter__(self) -> Iterator[T]:
        return self.future.heapsort()
