"""backfilling_scheduler - Module for a conservative backfilling scheduler"""

import copy as _copy

from schedgym.event import EventType
from schedgym.scheduler import Scheduler


class BackfillingScheduler(Scheduler):
    """Implements a conservative backfilling scheduler."""

    def schedule(self) -> None:
        self._schedule_bitmask()
        self.queue_admission.clear()
        self._queued_work_total = 0

    def _schedule_bitmask(self) -> None:
        from schedgym.scheduler._bitmask_timeline import (
            build_timeline,
            find_earliest_fit,
            update_timeline,
            bitmask_to_resource,
        )

        future_by_time, future_times = self.job_events.future_events_snapshot()

        base_memory_pool = None
        total_memory = 0
        scheduled_allocations: list[tuple[int, int, int, list]] = []

        if not self.ignore_memory:
            base_memory_pool = _copy.copy(self.cluster.memory.used_pool)
            total_memory = self.cluster.memory.size

        timeline = build_timeline(
            self.current_time,
            self.number_of_processors,
            self.cluster.processors.used_pool,
            future_by_time,
            future_times,
            used_memory_pool=base_memory_pool if not self.ignore_memory else None,
            total_memory=total_memory,
        )

        for job in self.queue_admission:
            start_time, allocated_mask = find_earliest_fit(
                timeline, job, ignore_memory=self.ignore_memory
            )

            if not self.ignore_memory and base_memory_pool is not None:
                memory_pool_at_start = _copy.copy(base_memory_pool)

                scheduled_ids = {alloc[2] for alloc in scheduled_allocations}
                for t in sorted(future_by_time.keys()):
                    if t > start_time:
                        break
                    for ev in future_by_time[t]:
                        if ev.job.id in scheduled_ids:
                            continue
                        if ev.type == EventType.JOB_FINISH:
                            for iv in ev.job.resources.memory:
                                try:
                                    memory_pool_at_start.chop(iv.begin, iv.end)
                                except Exception:
                                    pass
                        elif ev.type == EventType.JOB_START and t < start_time:
                            for iv in ev.job.resources.memory:
                                memory_pool_at_start.add(iv)

                for (
                    alloc_start,
                    alloc_finish,
                    alloc_job_id,
                    alloc_ivs,
                ) in scheduled_allocations:
                    if alloc_start <= start_time < alloc_finish:
                        for iv in alloc_ivs:
                            memory_pool_at_start.add(iv)
            else:
                memory_pool_at_start = None

            resources = bitmask_to_resource(
                allocated_mask,
                job.id,
                memory_amount=job.requested_memory if not self.ignore_memory else 0,
                total_memory=total_memory,
                memory_used_pool=memory_pool_at_start,
            )
            if not resources:
                raise AssertionError("Something is terribly wrong")
            self.assign_schedule(job, resources, start_time)
            update_timeline(
                timeline,
                job,
                start_time,
                allocated_mask,
                memory_amount=job.requested_memory if not self.ignore_memory else 0,
            )
            if not self.ignore_memory:
                finish_time = start_time + job.execution_time
                scheduled_allocations.append(
                    (start_time, finish_time, job.id, list(resources.memory))
                )
