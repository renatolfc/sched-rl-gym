from dataclasses import dataclass
from pathlib import Path

from schedgym.scheduler import Scheduler


@dataclass(frozen=True)
class ReplayConfig:
    scheduler_cls: type[Scheduler]
    processors: int
    memory: int
    ignore_memory: bool = True
    trace_limit: int | None = None
    timeout_s: float | None = None
    verify_monotonic_time: bool = True
    trace_path: str | Path | None = None
