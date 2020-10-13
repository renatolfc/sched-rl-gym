#!/usr/bin/env python
# -*- coding: utf-8 -*-

"random - a random scheduler"

import random

from lugarrl.scheduler import PackerScheduler


class RandomScheduler(PackerScheduler):
    """A random scheduling policy.

    This reuses functionality from the :class:`PackerScheduler`. Therefore, it
    only needs to define a random priority function.
    """
    def get_priority(self, _) -> int:
        "Random priority function for random scheduler."
        return random.randint(0, len(self.queue_admission) - 1)
