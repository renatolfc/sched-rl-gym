#!/usr/bin/env python
# -*- coding: utf-8 -*-

"base - base module for all workload generators"

from abc import ABC, abstractmethod
from typing import Optional, List

from schedgym.job import Job


class WorkloadGenerator(ABC):
    "An abstract workload generator"
    current_time: int

    @abstractmethod
    def step(self, offset: int = 1) -> List[Optional[Job]]:
        """Steps the workload generator by :param offset:.

        This may, or may not, return new jobs, depending on the internal
        probability distributions of the workload generator.

        Parameters
        ----------
            offset : int
                The number of time steps to advance the workload generator.
         """

    @abstractmethod
    def __len__(self):
        """Returns the length of the workload. Zero if unbounded."""

    @abstractmethod
    def peek(self):
        """Peeks what would be the next job"""
