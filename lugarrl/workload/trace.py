#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import takewhile
from typing import Iterator, Optional, Sequence

from ..job import Job
from .base import WorkloadGenerator
from .swf_parser import parse as parse_swf


class TraceGenerator(WorkloadGenerator):
    restart: bool
    tracefile: str
    trace: Sequence[Job]

    def sample(self, submission_time=-1):
        if submission_time < 0:
            raise ValueError('Submission time must not be negative')
        if submission_time < self.last_event_time:
            raise ValueError('Submission time cannot be smaller than previous time')
        jobs = takewhile(lambda j: j[1].submission_time <= submission_time,
                         enumerate(self.trace[self.current_element:], self.current_element))
        jobs = list(jobs)
        if jobs:
            self.current_element = jobs[-1][0] + 1
            return [j for (i, j) in jobs]
        else:
            return []

    def __init__(self, tracefile, processors, memory,
                 offset=0, length=None, restart=False):
        self.restart = restart
        self.tracefile = tracefile
        self.trace = [j for j in parse_swf(tracefile, processors, memory)]

        if length is None:
            length = len(self.trace)
        else:
            length = length if length <= len(self.trace) else len(self.trace)

        self.trace = self.trace[offset:offset+length]

        self.current_element = 0

    @property
    def last_event_time(self):
        offset = self.current_element if self.current_element < len(self.trace) else -1
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
