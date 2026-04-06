"""base - base module for all workload generators"""

from abc import ABC, abstractmethod

from schedgym.job import Job


class WorkloadGenerator(ABC):
    """An abstract workload generator"""

    current_time: int

    @abstractmethod
    def step(self, offset: int = 1) -> list[Job | None]:
        """Steps the workload generator by :param offset:.

        This may, or may not, return new jobs, depending on the internal
        probability distributions of the workload generator.

        Parameters
        ----------
            offset : int
                The number of time steps to advance the workload generator.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the workload. Zero if unbounded."""

    @abstractmethod
    def peek(self) -> Job | None:
        """Peeks what would be the next job"""
