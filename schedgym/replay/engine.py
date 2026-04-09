from __future__ import annotations

import time
from pathlib import Path

from schedgym.job import Job
from schedgym.replay.config import ReplayConfig
from schedgym.replay.result import ReplayResult
from schedgym.replay.trace_cursor import TraceCursor
from schedgym.scheduler import (
    BackfillingScheduler,
    EasyScheduler,
    FifoScheduler,
    Scheduler,
)
from schedgym.workload.swf_parser import parse as parse_swf


class TraceReplayEngine:
    def __init__(self, jobs: list[Job], config: ReplayConfig) -> None:
        supported = (
            FifoScheduler,
            EasyScheduler,
            BackfillingScheduler,
        )
        if not issubclass(config.scheduler_cls, supported):
            raise NotImplementedError(
                "Replay currently supports only FifoScheduler, EasyScheduler, and BackfillingScheduler (or subclasses)"
            )
        self.jobs = jobs
        self.config = config
        self.cursor = TraceCursor(jobs)
        if config.verify_monotonic_time:
            self.cursor.validate_monotonic()
        self.scheduler: Scheduler = config.scheduler_cls(
            config.processors,
            config.memory,
            ignore_memory=config.ignore_memory,
        )
        self.jobs_submitted = 0
        self.events_processed = 0
        self.jobs_dropped_before_engine = 0
        self.source_jobs_loaded = len(jobs)

    @classmethod
    def from_swf(
        cls,
        trace_path: str | Path,
        config: ReplayConfig,
    ) -> TraceReplayEngine:
        loaded_records = 0
        with open(trace_path, "r") as handle:
            for line in handle:
                if ";" in line or not line.strip():
                    continue
                loaded_records += 1

        parsed_jobs = list(
            parse_swf(
                trace_path,
                config.processors,
                config.memory,
                config.ignore_memory,
            )
        )
        jobs = parsed_jobs
        if config.trace_limit is not None:
            jobs = jobs[: config.trace_limit]
        merged_config = ReplayConfig(
            scheduler_cls=config.scheduler_cls,
            processors=config.processors,
            memory=config.memory,
            ignore_memory=config.ignore_memory,
            trace_limit=config.trace_limit,
            timeout_s=config.timeout_s,
            verify_monotonic_time=config.verify_monotonic_time,
            trace_path=trace_path,
        )
        engine = cls(jobs, merged_config)
        engine.source_jobs_loaded = len(parsed_jobs)
        engine.jobs_dropped_before_engine = max(0, loaded_records - len(parsed_jobs))
        return engine

    def _next_timestamp(self) -> int | None:
        next_arrival = self.cursor.next_time()
        next_event = (
            self.scheduler.job_events.next.time
            if self.scheduler.job_events.next
            else None
        )
        timestamps = [t for t in (next_arrival, next_event) if t is not None]
        if not timestamps:
            return None
        return min(timestamps)

    def _advance_to(self, target_time: int) -> None:
        if target_time < self.scheduler.current_time:
            raise AssertionError("Replay engine cannot move backwards in time")
        processed = self.scheduler.replay_advance_to(target_time)
        self.events_processed += processed

    def _submit_arrivals_at(self, timestamp: int) -> int:
        arrivals = self.cursor.pop_jobs_at(timestamp)
        if arrivals:
            self.scheduler.submit(arrivals)
            self.jobs_submitted += len(arrivals)
        return len(arrivals)

    def _schedule_round(self) -> None:
        if self.scheduler.queue_admission:
            self.scheduler.replay_schedule()
            self._advance_to(self.scheduler.current_time)

    def _has_pending_work(self) -> bool:
        return bool(
            self.cursor.has_next()
            or self.scheduler.jobs_in_system
            or self.scheduler.job_events.next
        )

    def _drain(self) -> None:
        while self.scheduler.job_events.next is not None:
            next_event_time = self.scheduler.job_events.next.time
            self._advance_to(next_event_time)
            self._schedule_round()

    def _build_result(self, wall_time_s: float, timeout_hit: bool) -> ReplayResult:
        completed = self.scheduler.queue_completed
        avg_slowdown = (
            sum(job.slowdown for job in completed) / len(completed)
            if completed
            else None
        )
        avg_bounded_slowdown = (
            sum(job.bounded_slowdown for job in completed) / len(completed)
            if completed
            else None
        )
        trace_name = (
            str(self.config.trace_path) if self.config.trace_path else "in-memory-trace"
        )
        malformed_jobs_dropped = self.jobs_dropped_before_engine
        return ReplayResult(
            trace_name=trace_name,
            scheduler_name=type(self.scheduler).__name__,
            jobs_loaded=self.jobs_loaded,
            jobs_submitted=self.jobs_submitted,
            jobs_completed=len(completed),
            malformed_jobs_dropped=malformed_jobs_dropped,
            simulated_end_time=self.scheduler.current_time,
            makespan=self.scheduler.makespan,
            avg_slowdown=avg_slowdown,
            avg_bounded_slowdown=avg_bounded_slowdown,
            wall_time_s=wall_time_s,
            timeout_hit=timeout_hit,
            event_count_processed=self.events_processed,
        )

    @property
    def jobs_loaded(self) -> int:
        return len(self.jobs)

    def run(self) -> ReplayResult:
        start = time.perf_counter()
        timeout_hit = False
        while self._has_pending_work():
            if (
                self.config.timeout_s is not None
                and time.perf_counter() - start > self.config.timeout_s
            ):
                timeout_hit = True
                break
            next_timestamp = self._next_timestamp()
            if next_timestamp is None:
                break
            self._advance_to(next_timestamp)
            self._submit_arrivals_at(next_timestamp)
            self._schedule_round()

        if not timeout_hit:
            while self.scheduler.job_events.next is not None:
                if (
                    self.config.timeout_s is not None
                    and time.perf_counter() - start > self.config.timeout_s
                ):
                    timeout_hit = True
                    break
                next_event_time = self.scheduler.job_events.next.time
                self._advance_to(next_event_time)
                self._schedule_round()

        if not timeout_hit:
            if (
                self.cursor.has_next()
                or self.scheduler.jobs_in_system
                or self.scheduler.job_events.next
            ):
                raise AssertionError(
                    "Replay finished without draining all pending work"
                )
        return self._build_result(time.perf_counter() - start, timeout_hit)
