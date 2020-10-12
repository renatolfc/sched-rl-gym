#!/usr/bin/env python
# -*- coding: utf-8 -*-

"base - base module for all workload generators"

from abc import ABC, abstractmethod


class WorkloadGenerator(ABC):
    "An abstract workload generator"
    @abstractmethod
    def __next__(self):
        "Next element in iterator."

    @abstractmethod
    def __iter__(self):
        "Iterator."

    @abstractmethod
    def sample(self, submission_time=0):
        "Sample a job with submission time equal to :param submission_time:."
