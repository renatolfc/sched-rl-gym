#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from lugarrl.scheduler import PackerScheduler


class RandomScheduler(PackerScheduler):
    def get_priority(self, _) -> int:
        return random.randint(0, len(self.queue_admission))
