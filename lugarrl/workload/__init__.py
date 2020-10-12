#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""workload - Package for generators of load for a cluster.

Supports generative workloads, based on probability distributions, and
trace-based workloads in the Standard Workload Format.
"""

from .base import WorkloadGenerator
from .distribution import BinomialWorkloadGenerator
from .distribution import DistributionalWorkloadGenerator
from .trace import TraceGenerator

__all__ = [
    'WorkloadGenerator',
    'DistributionalWorkloadGenerator',
    'BinomialWorkloadGenerator',
    'TraceGenerator',
]
