"""workload - Package for generators of load for a cluster.

Supports generative workloads, based on probability distributions, and
trace-based workloads in the Standard Workload Format.
"""

from .base import WorkloadGenerator
from .distribution import BinomialWorkloadGenerator, DistributionalWorkloadGenerator
from .trace import SwfGenerator, TraceGenerator

__all__ = [
    "WorkloadGenerator",
    "DistributionalWorkloadGenerator",
    "BinomialWorkloadGenerator",
    "TraceGenerator",
    "SwfGenerator",
]
