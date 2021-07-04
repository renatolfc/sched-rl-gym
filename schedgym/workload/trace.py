#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trace - A trace-based workload generator

Inherits from the base WorkloadGenerator and uses the swf_parser to parse SWF
files.
"""

from itertools import takewhile
from typing import Iterator, Optional, Sequence, Callable

from ..job import Job
from .base import WorkloadGenerator
from .swf_parser import parse as parse_swf


class TraceGenerator(WorkloadGenerator):
    restart: bool
    trace: Sequence[Job]
    refresh_jobs: Optional[Callable] = None

    def __init__(self, restart=False, trace=None):
        self.current_time = 0
        self.restart = restart
        self.current_element = 0

        if trace is not None:
            self.trace = trace
        else:
            self.trace = []

    def step(self, offset=1):
        """"Samples" jobs from the trace file.

        Parameters
        ----------
            offset : int
                The amount to offset the current time step
        """
        if offset < 0:
            raise ValueError('Submission time must be positive')
        if self.current_element >= len(self.trace):
            if self.restart:
                self.current_element = 0
                for job in self.trace:
                    job.submission_time += self.current_time
                if self.refresh_jobs is not None:
                    self.refresh_jobs()
            else:
                raise StopIteration('Workload finished')
        submission_time = self.current_time + offset
        jobs = takewhile(
            lambda j: j[1].submission_time <= submission_time,
            enumerate(
                self.trace[self.current_element:], self.current_element
            )
        )
        self.current_time = submission_time
        jobs = list(jobs)
        if jobs:
            self.current_element = jobs[-1][0] + 1
            return [j for (i, j) in jobs]
        return []

    @property
    def last_event_time(self):
        "The submission time of the last generated job"
        offset = self.current_element \
            if self.current_element < len(self.trace) else -1
        return self.trace[offset].submission_time

    def __len__(self):
        return len(self.trace)

    def __next__(self) -> Job:
        if self.current_element >= len(self.trace):
            if self.restart:
                self.current_element = 0
                if self.refresh_jobs is not None:
                    self.refresh_jobs()
            else:
                raise StopIteration()
        job = self.trace[self.current_element]
        self.current_element += 1
        return job

    def __iter__(self) -> Iterator[Optional[Job]]:
        return iter(self.trace)

    def peek(self) -> Optional[Job]:
        job = next(self)
        if self.current_element > 0:
            self.current_element -= 1
        return job


class SwfGenerator(TraceGenerator):
    """A trace-based (workload log) generator.

    Supports starting the parsing after an offset, and also supports reading a
    pre-specified number of jobs.

    Parameters
    ----------
        tracefile : str
            The path to the filed to be parsed and used as input for workload
            generation.
        processors : int
            The number of processors in this trace
        memory : int
            The amount of memory in this trace
        restart : bool
            Whether to restart from the beginning of the file when we reach
            its end (or, in the case we're using an offset and a length, to
            restart from the offset up to the length)
        ignore_memory : bool
            Whether to ignore (or not) memory usage
    """

    tracefile: str
    ignore_memory: bool

    def __init__(self, tracefile, processors, memory,
                 offset=0, length=None, restart=False,
                 ignore_memory=False):

        super().__init__(
            restart,
            list(parse_swf(tracefile, processors, memory, ignore_memory))
        )
        self.tracefile = tracefile

        if length is None:
            length = len(self.trace)
        else:
            length = length if length <= len(self.trace) else len(self.trace)

        self.trace = self.trace[offset:offset+length]

        self.current_element = 0
