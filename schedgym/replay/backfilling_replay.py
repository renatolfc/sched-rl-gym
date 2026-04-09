"""Compatibility alias — BackfillingScheduler now has the bitmask path built in.

Deprecated: use BackfillingScheduler directly.
"""

from __future__ import annotations

from schedgym.scheduler.backfilling_scheduler import BackfillingScheduler


class BackfillingReplayScheduler(BackfillingScheduler):
    pass
