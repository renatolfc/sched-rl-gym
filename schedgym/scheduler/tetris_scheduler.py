#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tetris_scheduler - A scheduler that mixes Packer and SJF"""

from schedgym.job import Job
from schedgym.scheduler import PackerScheduler


class TetrisScheduler(PackerScheduler):
    """Implements the Tetris scheduler.

    Adds a factor that controls how much Packer behavior and how much SJF
    behavior influences the scheduler.

    Parameters
    ----------
        number_of_processors : int
            Number of processors in the cluster this scheduler manages
        total_memory : int
            Amount of memory in the cluster this scheduler manages
        packer_sjf_ratio : float
            Dial to tune packer to sjf ratio. Valid values in [0, 1], with
            0 being SJF and 1 being Packer behavior.
    """

    packer_sjf_ratio: float

    def __init__(
        self,
        number_of_processors: int,
        total_memory: int,
        packer_sjf_ratio: float = 0.5,
    ):
        super().__init__(number_of_processors, total_memory)
        self.packer_sjf_ratio = packer_sjf_ratio

    def get_priority(self, j: Job) -> float:
        """Gives the packer/sjf mixed priority.

        Parameters
        ----------
            j : Job
                The job for which we're computing priority.
        """
        return (
            self.packer_sjf_ratio
            * (
                self.free_resources[0] * j.requested_processors
                + self.free_resources[1]
                + j.requested_memory
            )
            + (1 - self.packer_sjf_ratio) * 1.0 / j.requested_time
        )
