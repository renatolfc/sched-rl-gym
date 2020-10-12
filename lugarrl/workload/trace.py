#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trace - A trace-based workload generator

Inherits from the base WorkloadGenerator and uses the swf_parser to parse SWF
files.
"""

from itertools import takewhile
from typing import Iterator, Optional, Sequence

from ..job import Job
from .base import WorkloadGenerator
from .swf_parser import parse as parse_swf


class TraceGenerator(WorkloadGenerator):
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
    restart: bool
    tracefile: str
    ignore_memory: bool
    trace: Sequence[Job]

    def __init__(self, tracefile, processors, memory,
                 offset=0, length=None, restart=False,
                 ignore_memory=False):
        self.restart = restart
        self.tracefile = tracefile
        self.trace = list(
            parse_swf(tracefile, processors, memory, ignore_memory)
        )

        if length is None:
            length = len(self.trace)
        else:
            length = length if length <= len(self.trace) else len(self.trace)

        self.trace = self.trace[offset:offset+length]

        self.current_element = 0

    def sample(self, submission_time=-1):
        """"Samples" jobs from the trace file.

        Parameters
        ----------
            submission_time : int
                The time at which to use to submit the job
        """
        if submission_time < 0:
            raise ValueError('Submission time must not be negative')
        if submission_time < self.last_event_time:
            raise ValueError(
                'Submission time cannot be smaller than previous time'
            )
        jobs = takewhile(
            lambda j: j[1].submission_time <= submission_time,
            enumerate(
                self.trace[self.current_element:], self.current_element
            )
        )
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
            else:
                raise StopIteration()
        job = self.trace[self.current_element]
        self.current_element += 1
        return job

    def __iter__(self) -> Iterator[Optional[Job]]:
        return iter(self.trace)
