#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""event - Event Handling classes

We have a basic Event type, which is specialized by
    1. A ResourceEvent, related to events that occur to resources and
    2. A JobEvent, related to events that occur to jobs
"""

import copy
import enum
import warnings
from typing import List, Optional, Iterable, TypeVar, Generic, Iterator

from .job import Job
from .heap import Heap
from .pool import ResourceType, Interval

T = TypeVar('T', bound='Event')  # pylint: disable=C
'Generic type for type annotations'


class EventType(enum.IntEnum):
    """Enumeration for the different types of events that can occur."""

    RESOURCE_ALLOCATE = 0
    RESOURCE_FREE = 1
    JOB_FINISH = 2
    JOB_START = 3


class Event:
    """A base event class.

    Parameters
    ----------
        time : int
            The time at which this event occurs
        type : EventType
            What is the type of this event
    """

    time: int
    type: EventType

    def __init__(self, time: int, type: EventType):
        # pylint: disable=redefined-builtin
        self.time = time
        self.type = type

    def clone(self):
        """Clones this event.

        Returns:
            A new event identical to this one, but with no memory sharing.
        """
        return copy.copy(self)


class ResourceEvent(Event):
    """An event related to resource allocation or to the freeing of resources.

    Parameters
    ----------
        time : int
            The time at which this event occurs
        type : EventType
            What is the type of this event
        resources : Iterable[Interval]
            The resources that are being allocated/free'd by this event
    """

    resources: Iterable[Interval]
    resource_type: ResourceType

    def __init__(
        self,
        time: int,
        type: EventType,
        resource_type: ResourceType,
        resources: Iterable[Interval],
    ):
        # pylint: disable=redefined-builtin
        super().__init__(time, type)
        self.resources = resources
        self.resource_type = resource_type


class JobEvent(Event):
    """An event related to the start of finishing of jobs.

    Parameters
    ----------
        time : int
            The time at which this event occurs
        type : EventType
            What is the type of this event
        job : Job
            The job to which this event applies
    """

    job: Job

    def __init__(self, time: int, type: EventType, job: Job):
        # pylint: disable=redefined-builtin
        super().__init__(time, type)
        self.job = job

    @property
    def processors(self) -> Iterable[Interval]:
        """The processors touched by the job that caused this event"""
        return self.job.resources.processors

    @property
    def memory(self) -> Iterable[Interval]:
        """The memory touched by the job that caused this event"""
        return self.job.resources.memory

    def __str__(self):
        return f'JobEvent<{self.time}, {self.type.name}, {self.job}>'

    def __repr__(self):
        return str(self)


class EventQueue(Generic[T]):
    """A priority-queue of events sorted by time.

    Parameters
    ----------
        time : int
            The moment in time this event queue begins.
    """

    time: int
    past: List[T]
    future: Heap[T]

    def __init__(self, time: int = 0):
        self.past = []
        self.time = time
        self.future = Heap()

    def add(self, event: T) -> None:
        """Adds a new event to the priority queue.

        Parameters
        ----------
            event
                The event to be added
        """
        if event.time >= self.time:
            self.future.add(event, (event.time, event.type))
        else:
            self.past.append(event)
            self.past.sort(key=lambda e: e.time)
            warnings.warn(
                'Adding events to the past might change the '
                'ordering of events that happened at the same '
                'time.'
            )

    def step(self, time: int = 1) -> Iterable[T]:
        """Steps time in the event queue.

        Parameters
        ----------
            time : int
                The amount of time steps to perform

        Returns:
            A list with all events that happened between the previous time and
            the current time.
        """
        if time < 0:
            raise AssertionError('Tried to move into the past.')
        self.time += time
        present: List[T] = []
        first = self.future.first
        while first and first.time <= self.time:
            current = self.future.pop()
            present.append(current)
            self.past.append(current)
            first = self.future.first
        return present

    def remove(self, event: Event) -> None:
        """Removes an event from the queue.

        The event is required to not have happened yet, as removal of past
        events is not supported.
        """
        if event not in self.future:
            raise ValueError('Tried to remove non-existant value')
        self.future.remove(event)

    @property
    def first(self) -> Optional[T]:  # XXX: This is probably not needed
        """The first event in the future to happen in this queue."""
        return self.future.first

    @property
    def next(self) -> Optional[T]:
        """The next event to happen in this queue."""
        if len(self.future) == 0:
            return None
        return self.future.first

    @property
    def last(self) -> Optional[T]:
        """The last event to have happened in this queue."""
        return self.past[-1] if self.past else None

    def __iter__(self) -> Iterator[T]:
        return self.future.heapsort()

    def __str__(self) -> str:
        return f'{[e for e in self.future.heapsort()]}'

    def __repr__(self):
        return str(self)
