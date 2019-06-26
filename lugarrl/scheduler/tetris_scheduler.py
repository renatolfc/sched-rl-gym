#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lugarrl.job import Job
from lugarrl.scheduler import PackerScheduler


class TetrisScheduler(PackerScheduler):
    """Implements the Tetris scheduler."""

    packer_sjf_ratio: float

    def __init__(self, number_of_processors, total_memory, packer_sjf_ratio: float = 0.5):
        super().__init__(number_of_processors, total_memory)
        self.packer_sjf_ratio = packer_sjf_ratio

    def get_priority(self, j: Job) -> int:
        return self.packer_sjf_ratio * (
            self.free_resources[0] * j.requested_processors + self.free_resources[1] + j.requested_memory
        ) + (1 - self.packer_sjf_ratio) * 1.0 / j.requested_time

