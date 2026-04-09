from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayResult:
    trace_name: str
    scheduler_name: str
    jobs_loaded: int
    jobs_submitted: int
    jobs_completed: int
    malformed_jobs_dropped: int
    simulated_end_time: int
    makespan: int
    avg_slowdown: float | None
    avg_bounded_slowdown: float | None
    wall_time_s: float
    timeout_hit: bool
    event_count_processed: int
