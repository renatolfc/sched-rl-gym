"""random - a random scheduler"""

import random

from schedgym.scheduler import PackerScheduler


class RandomScheduler(PackerScheduler):
    """A random scheduling policy.

    This reuses functionality from the :class:`PackerScheduler`. Therefore, it
    only needs to define a random priority function.
    """

    def get_priority(self, _) -> int:
        """Random priority function for random scheduler."""
        n = len(self.queue_admission)
        return random.randint(0, n - 1) if n > 0 else 0
