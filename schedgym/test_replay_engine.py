import gzip
import tempfile
import unittest
from pathlib import Path

from . import job, scheduler
from .job import Job
from .replay import BackfillingReplayScheduler, ReplayConfig, TraceReplayEngine
from .scheduler import Scheduler
from .scheduler.backfilling_scheduler import BackfillingScheduler


def build_job(job_id: int, submission: int, duration: int, processors: int) -> Job:
    params = job.JobParameters(duration, duration, processors, processors, 1, 1)
    built = params.sample(submission)
    built.id = job_id
    built.execution_time = duration
    built.requested_time = duration
    built.requested_processors = processors
    built.processors_allocated = processors
    built.memory_use = 1
    built.requested_memory = 1
    return built


def run_tick_scheduler(
    jobs: list[Job], processors: int, memory: int, scheduler_cls: type[Scheduler]
) -> Scheduler:
    sched = scheduler_cls(processors, memory, ignore_memory=True)
    current_time = 0
    index = 0
    jobs = sorted(jobs, key=lambda job: (job.submission_time, job.id))

    while index < len(jobs) or sched.jobs_in_system or sched.job_events.next:
        current_time += 1
        sched.step()
        while index < len(jobs) and jobs[index].submission_time <= current_time:
            same_time = []
            while index < len(jobs) and jobs[index].submission_time <= current_time:
                same_time.append(jobs[index])
                index += 1
            sched.submit(same_time)
    return sched


def run_tick_scheduler_preserving_order(
    jobs: list[Job], processors: int, memory: int, scheduler_cls: type[Scheduler]
) -> Scheduler:
    sched = scheduler_cls(processors, memory, ignore_memory=True)
    current_time = 0
    index = 0

    while index < len(jobs) or sched.jobs_in_system or sched.job_events.next:
        current_time += 1
        sched.step()
        same_time = []
        while index < len(jobs) and jobs[index].submission_time <= current_time:
            same_time.append(jobs[index])
            index += 1
        if same_time:
            sched.submit(same_time)
    return sched


class TestReplayEngine(unittest.TestCase):
    def assert_equivalent(
        self, tick_sched: Scheduler, replay_engine: TraceReplayEngine
    ):
        replay_sched = replay_engine.scheduler
        self.assertEqual(tick_sched.makespan, replay_sched.makespan)
        self.assertEqual(
            len(tick_sched.queue_completed), len(replay_sched.queue_completed)
        )

        tick_completed = sorted(tick_sched.queue_completed, key=lambda job: job.id)
        replay_completed = sorted(replay_sched.queue_completed, key=lambda job: job.id)
        self.assertEqual(
            [job.id for job in tick_completed], [job.id for job in replay_completed]
        )
        self.assertEqual(
            [job.start_time for job in tick_completed],
            [job.start_time for job in replay_completed],
        )
        self.assertEqual(
            [job.finish_time for job in tick_completed],
            [job.finish_time for job in replay_completed],
        )

    def run_replay(
        self,
        jobs: list[Job],
        processors: int,
        memory: int,
        scheduler_cls: type[Scheduler] = scheduler.FifoScheduler,
    ):
        engine = TraceReplayEngine(
            jobs,
            ReplayConfig(
                scheduler_cls=scheduler_cls,
                processors=processors,
                memory=memory,
                ignore_memory=True,
            ),
        )
        result = engine.run()
        self.assertFalse(result.timeout_hit)
        return engine

    def test_single_job_equivalence(self):
        jobs = [build_job(1, 1, 2, 1)]
        tick = run_tick_scheduler(
            [build_job(1, 1, 2, 1)], 4, 4, scheduler.FifoScheduler
        )
        replay = self.run_replay(jobs, 4, 4)
        self.assert_equivalent(tick, replay)

    def test_same_time_arrivals_equivalence(self):
        jobs = [
            build_job(1, 1, 3, 1),
            build_job(2, 1, 2, 1),
            build_job(3, 1, 1, 1),
        ]
        tick = run_tick_scheduler(
            [build_job(1, 1, 3, 1), build_job(2, 1, 2, 1), build_job(3, 1, 1, 1)],
            4,
            4,
            scheduler.FifoScheduler,
        )
        replay = self.run_replay(jobs, 4, 4)
        self.assert_equivalent(tick, replay)

    def test_idle_gap_equivalence(self):
        jobs = [build_job(1, 100, 2, 1), build_job(2, 105, 2, 1)]
        tick = run_tick_scheduler(
            [build_job(1, 100, 2, 1), build_job(2, 105, 2, 1)],
            4,
            4,
            scheduler.FifoScheduler,
        )
        replay = self.run_replay(jobs, 4, 4)
        self.assert_equivalent(tick, replay)

    def test_fifo_blocking_equivalence(self):
        jobs = [
            build_job(1, 1, 5, 3),
            build_job(2, 1, 1, 1),
            build_job(3, 1, 1, 1),
        ]
        tick = run_tick_scheduler(
            [build_job(1, 1, 5, 3), build_job(2, 1, 1, 1), build_job(3, 1, 1, 1)],
            3,
            3,
            scheduler.FifoScheduler,
        )
        replay = self.run_replay(jobs, 3, 3)
        self.assert_equivalent(tick, replay)

    def test_easy_replay_equivalence(self):
        jobs = [
            build_job(1, 0, 2, 2),
            build_job(2, 1, 2, 1),
            build_job(3, 1, 2, 3),
            build_job(4, 1, 1, 2),
            build_job(5, 1, 1, 2),
            build_job(6, 3, 1, 2),
        ]
        tick = run_tick_scheduler(
            [
                build_job(1, 0, 2, 2),
                build_job(2, 1, 2, 1),
                build_job(3, 1, 2, 3),
                build_job(4, 1, 1, 2),
                build_job(5, 1, 1, 2),
                build_job(6, 3, 1, 2),
            ],
            3,
            3,
            scheduler.EasyScheduler,
        )
        replay = self.run_replay(jobs, 3, 3, scheduler.EasyScheduler)
        self.assert_equivalent(tick, replay)

    def test_backfilling_replay_equivalence(self):
        jobs = [
            build_job(1, 0, 4, 2),
            build_job(2, 1, 2, 1),
            build_job(3, 1, 3, 2),
            build_job(4, 2, 1, 1),
        ]
        tick = run_tick_scheduler(
            [
                build_job(1, 0, 4, 2),
                build_job(2, 1, 2, 1),
                build_job(3, 1, 3, 2),
                build_job(4, 2, 1, 1),
            ],
            3,
            3,
            scheduler.BackfillingScheduler,
        )
        replay = self.run_replay(jobs, 3, 3, scheduler.BackfillingScheduler)
        self.assert_equivalent(tick, replay)

    def test_easy_same_timestamp_finish_and_arrival_equivalence(self):
        jobs = [
            build_job(1, 0, 2, 2),
            build_job(2, 2, 1, 2),
            build_job(3, 2, 1, 1),
        ]
        tick = run_tick_scheduler(
            [build_job(1, 0, 2, 2), build_job(2, 2, 1, 2), build_job(3, 2, 1, 1)],
            2,
            2,
            scheduler.EasyScheduler,
        )
        replay = self.run_replay(jobs, 2, 2, scheduler.EasyScheduler)
        self.assert_equivalent(tick, replay)

    def test_replay_preserves_input_order_for_same_time_jobs(self):
        jobs = [
            build_job(3, 1, 1, 1),
            build_job(1, 1, 1, 1),
            build_job(2, 1, 1, 1),
        ]
        tick = run_tick_scheduler_preserving_order(
            [build_job(3, 1, 1, 1), build_job(1, 1, 1, 1), build_job(2, 1, 1, 1)],
            4,
            4,
            scheduler.FifoScheduler,
        )
        replay = self.run_replay(jobs, 4, 4, scheduler.FifoScheduler)
        self.assert_equivalent(tick, replay)

    def test_replay_timeout(self):
        jobs = [build_job(i, i, 1, 1) for i in range(1, 1000)]
        engine = TraceReplayEngine(
            jobs,
            ReplayConfig(
                scheduler_cls=scheduler.FifoScheduler,
                processors=4,
                memory=4,
                ignore_memory=True,
                timeout_s=0.0,
            ),
        )
        result = engine.run()
        self.assertTrue(result.timeout_hit)

    def test_replay_rejects_non_fifo_scheduler(self):
        with self.assertRaises(NotImplementedError):
            TraceReplayEngine(
                [build_job(1, 1, 1, 1)],
                ReplayConfig(
                    scheduler_cls=scheduler.SjfScheduler,
                    processors=4,
                    memory=4,
                    ignore_memory=True,
                ),
            )

    def test_replay_rejects_non_monotonic_trace(self):
        jobs = [build_job(1, 2, 1, 1), build_job(2, 1, 1, 1)]
        with self.assertRaises(ValueError):
            TraceReplayEngine(
                jobs,
                ReplayConfig(
                    scheduler_cls=scheduler.FifoScheduler,
                    processors=4,
                    memory=4,
                    ignore_memory=True,
                    verify_monotonic_time=True,
                ),
            )

    def test_from_swf_runs_small_trace(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".swf", delete=False
        ) as handle:
            handle.write("1 1 0 2 1 1.0 1 1 2 1 1 1 1 1 1 1 -1 -1\n")
            handle.write("2 3 0 1 1 1.0 1 1 1 1 1 1 1 1 1 1 -1 -1\n")
            path = Path(handle.name)
        try:
            engine = TraceReplayEngine.from_swf(
                path,
                ReplayConfig(
                    scheduler_cls=scheduler.FifoScheduler,
                    processors=4,
                    memory=4,
                    ignore_memory=True,
                    trace_limit=2,
                ),
            )
            result = engine.run()
            self.assertEqual(2, result.jobs_completed)
            self.assertFalse(result.timeout_hit)
        finally:
            path.unlink(missing_ok=True)

    def test_from_gzipped_swf_runs_small_trace(self):
        with tempfile.NamedTemporaryFile(suffix=".swf.gz", delete=False) as handle:
            path = Path(handle.name)
        try:
            with gzip.open(path, "wt", encoding="utf-8") as trace:
                trace.write("1 1 0 2 1 1.0 1 1 2 1 1 1 1 1 1 1 -1 -1\n")
            engine = TraceReplayEngine.from_swf(
                path,
                ReplayConfig(
                    scheduler_cls=scheduler.FifoScheduler,
                    processors=4,
                    memory=4,
                    ignore_memory=True,
                ),
            )

            result = engine.run()

            self.assertEqual(1, result.jobs_completed)
            self.assertFalse(result.timeout_hit)
        finally:
            path.unlink(missing_ok=True)


def build_job_with_wallclock(
    job_id: int,
    submission: int,
    execution_time: int,
    requested_time: int,
    processors: int,
) -> Job:
    params = job.JobParameters(
        execution_time, execution_time, processors, processors, 1, 1
    )
    built = params.sample(submission)
    built.id = job_id
    built.execution_time = execution_time
    built.requested_time = requested_time
    built.requested_processors = processors
    built.processors_allocated = processors
    built.memory_use = 1
    built.requested_memory = 1
    return built


class TestBackfillingReplayScheduler(unittest.TestCase):
    def assert_equivalent(
        self, tick_sched: Scheduler, replay_engine: TraceReplayEngine
    ):
        replay_sched = replay_engine.scheduler
        self.assertEqual(tick_sched.makespan, replay_sched.makespan)
        self.assertEqual(
            len(tick_sched.queue_completed), len(replay_sched.queue_completed)
        )

        tick_completed = sorted(tick_sched.queue_completed, key=lambda j: j.id)
        replay_completed = sorted(replay_sched.queue_completed, key=lambda j: j.id)
        self.assertEqual(
            [j.id for j in tick_completed], [j.id for j in replay_completed]
        )
        self.assertEqual(
            [j.start_time for j in tick_completed],
            [j.start_time for j in replay_completed],
        )
        self.assertEqual(
            [j.finish_time for j in tick_completed],
            [j.finish_time for j in replay_completed],
        )

    def run_both(self, jobs_fn, processors, memory):
        tick = run_tick_scheduler(
            jobs_fn(), processors, memory, scheduler.BackfillingScheduler
        )
        engine = TraceReplayEngine(
            jobs_fn(),
            ReplayConfig(
                scheduler_cls=BackfillingReplayScheduler,
                processors=processors,
                memory=memory,
                ignore_memory=True,
            ),
        )
        result = engine.run()
        self.assertFalse(result.timeout_hit)
        self.assert_equivalent(tick, engine)
        return tick, engine

    def test_basic_backfilling_equivalence(self):
        def jobs():
            return [
                build_job(1, 0, 4, 2),
                build_job(2, 1, 2, 1),
                build_job(3, 1, 3, 2),
                build_job(4, 2, 1, 1),
            ]

        self.run_both(jobs, 3, 3)

    def test_same_timestamp_finish_and_arrival(self):
        def jobs():
            return [
                build_job(1, 0, 3, 2),
                build_job(2, 3, 2, 2),
                build_job(3, 3, 1, 1),
            ]

        self.run_both(jobs, 2, 2)

    def test_job_scheduled_at_current_time(self):
        def jobs():
            return [
                build_job(1, 0, 2, 1),
                build_job(2, 0, 2, 1),
                build_job(3, 0, 2, 1),
                build_job(4, 0, 2, 1),
            ]

        self.run_both(jobs, 4, 4)

    def test_window_boundary_exact_span(self):
        def jobs():
            return [
                build_job(1, 0, 5, 2),
                build_job(2, 1, 4, 2),
                build_job(3, 2, 1, 1),
            ]

        self.run_both(jobs, 3, 3)

    def test_requested_time_differs_from_execution_time(self):
        def jobs():
            return [
                build_job_with_wallclock(
                    1, 0, execution_time=3, requested_time=5, processors=2
                ),
                build_job_with_wallclock(
                    2, 1, execution_time=1, requested_time=4, processors=2
                ),
                build_job_with_wallclock(
                    3, 1, execution_time=2, requested_time=2, processors=1
                ),
                build_job_with_wallclock(
                    4, 2, execution_time=1, requested_time=1, processors=1
                ),
            ]

        self.run_both(jobs, 3, 3)

    def test_many_jobs_tight_cluster(self):
        def jobs():
            return [
                build_job(1, 0, 10, 2),
                build_job(2, 0, 5, 1),
                build_job(3, 1, 3, 2),
                build_job(4, 2, 2, 1),
                build_job(5, 3, 4, 3),
                build_job(6, 4, 1, 1),
                build_job(7, 5, 2, 2),
                build_job(8, 6, 3, 1),
            ]

        self.run_both(jobs, 3, 3)

    def test_counts_fit_but_no_common_cpu_across_window(self):
        def jobs():
            return [
                build_job(1, 0, 4, 1),
                build_job(2, 0, 2, 1),
                build_job(3, 1, 3, 2),
            ]

        self.run_both(jobs, 2, 2)

    def test_single_processor_cluster(self):
        def jobs():
            return [
                build_job(1, 0, 3, 1),
                build_job(2, 1, 2, 1),
                build_job(3, 2, 1, 1),
            ]

        self.run_both(jobs, 1, 1)

    def test_all_jobs_same_submission_time(self):
        def jobs():
            return [
                build_job(1, 0, 3, 2),
                build_job(2, 0, 2, 1),
                build_job(3, 0, 4, 2),
                build_job(4, 0, 1, 1),
            ]

        self.run_both(jobs, 3, 3)

    def test_larger_scenario_with_mixed_durations(self):
        def jobs():
            result = []
            for i in range(1, 21):
                dur = (i * 7 + 3) % 10 + 1
                procs = (i * 3) % 4 + 1
                sub = (i - 1) * 2
                result.append(build_job(i, sub, dur, procs))
            return result

        self.run_both(jobs, 8, 8)


class _ForceIntervalTreeScheduler(BackfillingScheduler):
    """Forces the IntervalTree path even when ignore_memory=True."""

    def schedule(self) -> None:
        for queued_job in self.queue_admission:
            time, resources = self.find_first_time_for(queued_job)
            if not resources:
                raise AssertionError("Something is terribly wrong")
            self.assign_schedule(queued_job, resources, time)
        self.queue_admission.clear()


def build_job_with_memory(
    job_id: int, submission: int, duration: int, processors: int, memory: int
) -> Job:
    params = job.JobParameters(
        duration, duration, processors, processors, memory, memory
    )
    built = params.sample(submission)
    built.id = job_id
    built.execution_time = duration
    built.requested_time = duration
    built.requested_processors = processors
    built.processors_allocated = processors
    built.memory_use = memory
    built.requested_memory = memory
    return built


class _ForceIntervalTreeSchedulerWithMemory(BackfillingScheduler):
    def __init__(self, number_of_processors, total_memory, ignore_memory=False):
        super().__init__(number_of_processors, total_memory, ignore_memory=False)

    def schedule(self) -> None:
        for queued_job in self.queue_admission:
            time, resources = self.find_first_time_for(queued_job)
            if not resources:
                raise AssertionError("Something is terribly wrong")
            self.assign_schedule(queued_job, resources, time)
        self.queue_admission.clear()


class _ForceBitmaskSchedulerWithMemory(BackfillingScheduler):
    def __init__(self, number_of_processors, total_memory, ignore_memory=False):
        super().__init__(number_of_processors, total_memory, ignore_memory=False)

    def schedule(self) -> None:
        self._schedule_bitmask()
        self.queue_admission.clear()


class TestBitmaskVsIntervalTreeEquivalence(unittest.TestCase):
    def _compare_paths(self, jobs_fn, processors, memory):
        bitmask_engine = TraceReplayEngine(
            jobs_fn(),
            ReplayConfig(
                scheduler_cls=BackfillingScheduler,
                processors=processors,
                memory=memory,
                ignore_memory=True,
            ),
        )
        bitmask_result = bitmask_engine.run()
        self.assertFalse(bitmask_result.timeout_hit)

        intervaltree_engine = TraceReplayEngine(
            jobs_fn(),
            ReplayConfig(
                scheduler_cls=_ForceIntervalTreeScheduler,
                processors=processors,
                memory=memory,
                ignore_memory=True,
            ),
        )
        intervaltree_result = intervaltree_engine.run()
        self.assertFalse(intervaltree_result.timeout_hit)

        bitmask_sched = bitmask_engine.scheduler
        it_sched = intervaltree_engine.scheduler

        self.assertEqual(bitmask_sched.makespan, it_sched.makespan)
        self.assertEqual(
            len(bitmask_sched.queue_completed), len(it_sched.queue_completed)
        )

        bm_completed = sorted(bitmask_sched.queue_completed, key=lambda j: j.id)
        it_completed = sorted(it_sched.queue_completed, key=lambda j: j.id)
        self.assertEqual([j.id for j in bm_completed], [j.id for j in it_completed])
        self.assertEqual(
            [j.start_time for j in bm_completed],
            [j.start_time for j in it_completed],
        )

    def test_bitmask_vs_intervaltree_basic(self):
        def jobs():
            return [
                build_job(1, 0, 4, 2),
                build_job(2, 1, 2, 1),
                build_job(3, 1, 3, 2),
                build_job(4, 2, 1, 1),
            ]

        self._compare_paths(jobs, 3, 3)

    def test_bitmask_vs_intervaltree_requested_ne_execution(self):
        def jobs():
            return [
                build_job_with_wallclock(
                    1, 0, execution_time=3, requested_time=5, processors=2
                ),
                build_job_with_wallclock(
                    2, 1, execution_time=1, requested_time=4, processors=2
                ),
                build_job_with_wallclock(
                    3, 1, execution_time=2, requested_time=2, processors=1
                ),
                build_job_with_wallclock(
                    4, 2, execution_time=1, requested_time=1, processors=1
                ),
            ]

        self._compare_paths(jobs, 3, 3)

    def test_bitmask_vs_intervaltree_many_jobs(self):
        def jobs():
            result = []
            for i in range(1, 21):
                dur = (i * 7 + 3) % 10 + 1
                procs = (i * 3) % 4 + 1
                sub = (i - 1) * 2
                result.append(build_job(i, sub, dur, procs))
            return result

        self._compare_paths(jobs, 8, 8)

    def test_bitmask_vs_intervaltree_tight_cluster(self):
        def jobs():
            return [
                build_job(1, 0, 10, 2),
                build_job(2, 0, 5, 1),
                build_job(3, 1, 3, 2),
                build_job(4, 2, 2, 1),
                build_job(5, 3, 4, 3),
                build_job(6, 4, 1, 1),
                build_job(7, 5, 2, 2),
                build_job(8, 6, 3, 1),
            ]

        self._compare_paths(jobs, 3, 3)


class TestBitmaskVsIntervalTreeMemoryEquivalence(unittest.TestCase):
    def _compare_paths_memory(self, jobs_fn, processors, memory):
        bitmask_engine = TraceReplayEngine(
            jobs_fn(),
            ReplayConfig(
                scheduler_cls=_ForceBitmaskSchedulerWithMemory,
                processors=processors,
                memory=memory,
                ignore_memory=False,
            ),
        )
        bitmask_result = bitmask_engine.run()
        self.assertFalse(bitmask_result.timeout_hit)

        intervaltree_engine = TraceReplayEngine(
            jobs_fn(),
            ReplayConfig(
                scheduler_cls=_ForceIntervalTreeSchedulerWithMemory,
                processors=processors,
                memory=memory,
                ignore_memory=False,
            ),
        )
        intervaltree_result = intervaltree_engine.run()
        self.assertFalse(intervaltree_result.timeout_hit)

        bitmask_sched = bitmask_engine.scheduler
        it_sched = intervaltree_engine.scheduler

        self.assertEqual(bitmask_sched.makespan, it_sched.makespan)
        self.assertEqual(
            len(bitmask_sched.queue_completed), len(it_sched.queue_completed)
        )

        bm_completed = sorted(bitmask_sched.queue_completed, key=lambda j: j.id)
        it_completed = sorted(it_sched.queue_completed, key=lambda j: j.id)
        self.assertEqual([j.id for j in bm_completed], [j.id for j in it_completed])
        self.assertEqual(
            [j.start_time for j in bm_completed],
            [j.start_time for j in it_completed],
        )
        self.assertEqual(
            [j.processors_allocated for j in bm_completed],
            [j.processors_allocated for j in it_completed],
        )
        self.assertEqual(
            [j.resources.measure()[1] for j in bm_completed],
            [j.resources.measure()[1] for j in it_completed],
        )

    def test_memory_basic(self):
        def jobs():
            return [
                build_job_with_memory(1, 0, 4, 2, 2),
                build_job_with_memory(2, 1, 2, 1, 1),
                build_job_with_memory(3, 1, 3, 2, 3),
                build_job_with_memory(4, 2, 1, 1, 1),
            ]

        self._compare_paths_memory(jobs, 4, 8)

    def test_memory_fragmentation(self):
        def jobs():
            return [
                build_job_with_memory(1, 0, 10, 2, 4),
                build_job_with_memory(2, 0, 5, 1, 2),
                build_job_with_memory(3, 1, 3, 2, 2),
                build_job_with_memory(4, 2, 2, 1, 3),
                build_job_with_memory(5, 3, 4, 3, 1),
            ]

        self._compare_paths_memory(jobs, 4, 8)

    def test_memory_tight(self):
        def jobs():
            return [
                build_job_with_memory(1, 0, 4, 2, 4),
                build_job_with_memory(2, 1, 2, 1, 3),
                build_job_with_memory(3, 1, 3, 2, 2),
                build_job_with_memory(4, 2, 1, 1, 1),
            ]

        self._compare_paths_memory(jobs, 4, 6)

    def test_memory_zero(self):
        def jobs():
            return [
                build_job_with_memory(1, 0, 4, 2, 1),
                build_job_with_memory(2, 1, 2, 1, 1),
            ]

        self._compare_paths_memory(jobs, 4, 4)


def _run_tick_scheduler_easy(
    jobs: list[Job], processors: int, memory: int, ignore_memory: bool = True
) -> "scheduler.EasyScheduler":
    sched = scheduler.EasyScheduler(processors, memory, ignore_memory=ignore_memory)
    current_time = 0
    index = 0
    jobs = sorted(jobs, key=lambda j: (j.submission_time, j.id))

    while index < len(jobs) or sched.jobs_in_system or sched.job_events.next:
        current_time += 1
        sched.step()
        while index < len(jobs) and jobs[index].submission_time <= current_time:
            same_time = []
            while index < len(jobs) and jobs[index].submission_time <= current_time:
                same_time.append(jobs[index])
                index += 1
            sched.submit(same_time)
    return sched


class TestEasySchedulerEquivalence(unittest.TestCase):
    def _compare_easy_paths(
        self,
        jobs_fn,
        processors: int,
        memory: int,
        ignore_memory: bool = True,
    ) -> None:
        tick_sched = _run_tick_scheduler_easy(
            jobs_fn(), processors, memory, ignore_memory=ignore_memory
        )

        replay_engine = TraceReplayEngine(
            jobs_fn(),
            ReplayConfig(
                scheduler_cls=scheduler.EasyScheduler,
                processors=processors,
                memory=memory,
                ignore_memory=ignore_memory,
            ),
        )
        result = replay_engine.run()
        self.assertFalse(result.timeout_hit)

        replay_sched = replay_engine.scheduler

        self.assertEqual(tick_sched.makespan, replay_sched.makespan)
        self.assertEqual(
            len(tick_sched.queue_completed), len(replay_sched.queue_completed)
        )

        tick_completed = sorted(tick_sched.queue_completed, key=lambda j: j.id)
        replay_completed = sorted(replay_sched.queue_completed, key=lambda j: j.id)
        self.assertEqual(
            [j.id for j in tick_completed], [j.id for j in replay_completed]
        )
        self.assertEqual(
            [j.start_time for j in tick_completed],
            [j.start_time for j in replay_completed],
        )

    def test_easy_basic(self):
        def jobs():
            return [
                build_job(1, 0, 4, 2),
                build_job(2, 1, 2, 1),
                build_job(3, 1, 3, 2),
                build_job(4, 2, 1, 1),
            ]

        self._compare_easy_paths(jobs, 3, 3)

    def test_easy_with_reservation(self):
        def jobs():
            # Job 1 holds all 4 procs for 5 units; job 2 also needs 4 procs but
            # arrives at t=0 and can't start → EasyScheduler creates reservation
            # via find_first_time_for. Job 3 (2 procs) backfills around reservation.
            return [
                build_job(1, 0, 5, 4),
                build_job(2, 0, 3, 4),
                build_job(3, 1, 2, 2),
            ]

        self._compare_easy_paths(jobs, 4, 4)

    def test_easy_many_jobs(self):
        def jobs():
            result = []
            for i in range(1, 13):
                dur = (i * 5 + 2) % 8 + 1
                procs = (i * 3) % 3 + 1
                sub = (i - 1) * 3
                result.append(build_job(i, sub, dur, procs))
            return result

        self._compare_easy_paths(jobs, 4, 4)

    def test_easy_memory(self):
        def jobs():
            return [
                build_job_with_memory(1, 0, 4, 2, 3),
                build_job_with_memory(2, 0, 2, 2, 3),
                build_job_with_memory(3, 1, 3, 1, 2),
                build_job_with_memory(4, 2, 2, 1, 1),
            ]

        self._compare_easy_paths(jobs, 4, 6, ignore_memory=False)


class TestCanScheduleNowEquivalence(unittest.TestCase):
    def test_csn_fits_easily(self):
        sched = scheduler.FifoScheduler(4, 4)
        sched.submit(build_job(1, 0, 10, 2))
        sched.step()
        result = sched.can_schedule_now(build_job(100, sched.current_time, 5, 2))
        self.assertTrue(bool(result))

    def test_csn_doesnt_fit(self):
        sched = scheduler.FifoScheduler(4, 4)
        sched.submit([build_job(1, 0, 10, 2), build_job(2, 0, 10, 2)])
        sched.step()
        result = sched.can_schedule_now(build_job(100, sched.current_time, 5, 1))
        self.assertFalse(bool(result))

    def test_csn_fits_with_future_conflict(self):
        sched = scheduler.FifoScheduler(4, 4)
        sched.submit(build_job(1, 0, 10, 2))
        sched.step()
        future_job = build_job(2, sched.current_time, 8, 2)
        future_resources = sched.cluster.find(future_job)
        sched.assign_schedule(future_job, future_resources, sched.current_time + 1)
        result = sched.can_schedule_now(build_job(100, sched.current_time, 5, 2))
        self.assertFalse(bool(result))

    def test_csn_memory_constrained(self):
        sched = scheduler.FifoScheduler(4, 4, ignore_memory=False)
        sched.submit(build_job_with_memory(1, 0, 10, 2, 4))
        sched.step()
        result = sched.can_schedule_now(
            build_job_with_memory(100, sched.current_time, 5, 1, 1)
        )
        self.assertFalse(bool(result))

    def test_csn_exact_fit(self):
        sched = scheduler.FifoScheduler(4, 4)
        sched.submit(build_job(1, 0, 10, 2))
        sched.step()
        self.assertTrue(
            bool(sched.can_schedule_now(build_job(100, sched.current_time, 5, 2)))
        )
        self.assertFalse(
            bool(sched.can_schedule_now(build_job(101, sched.current_time, 5, 3)))
        )
