"""CPU reservation timeline using bitmasks per time segment.

This module replaces the IntervalTree-based resource search with exact
processor bitmask operations for conservative backfilling when
``ignore_memory=True``.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable

from schedgym.event import EventType, JobEvent
from schedgym.job import Job

try:
    from schedgym._schedgym_rs import Interval, IntervalTree
except ImportError:
    from intervaltree import Interval, IntervalTree

from schedgym.resource import Resource


class BitmaskTimeline:
    """CPU reservation timeline using bitmasks per time segment.

    Segment ``i`` covers the half-open interval ``[times[i], times[i+1])``.
    The last entry covers ``[times[-1], +inf)``.

    ``occupied[i]`` is a Python ``int`` bitmask where bit *j* is set iff
    processor *j* is occupied during that segment.
    """

    __slots__ = (
        "times",
        "occupied",
        "num_processors",
        "full_mask",
        "used_memory",
        "total_memory",
    )

    def __init__(
        self,
        times: list[int],
        occupied: list[int],
        num_processors: int,
        used_memory: list[int] | None = None,
        total_memory: int = 0,
    ) -> None:
        self.times = times
        self.occupied = occupied
        self.num_processors = num_processors
        self.full_mask = (1 << num_processors) - 1
        self.used_memory: list[int] = (
            used_memory if used_memory is not None else [0] * len(times)
        )
        self.total_memory: int = total_memory


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _intervaltree_to_bitmask(tree: IntervalTree) -> int:
    """Convert an IntervalTree of processor intervals to a bitmask."""
    mask = 0
    for iv in tree:
        if iv.end > iv.begin:
            mask |= ((1 << (iv.end - iv.begin)) - 1) << iv.begin
    return mask


def bitmask_to_intervals(mask: int, data: int | None = None) -> IntervalTree:
    """Convert a bitmask back to an IntervalTree of contiguous intervals."""
    tree = IntervalTree()
    if mask == 0:
        return tree
    pos = 0
    num_bits = mask.bit_length()
    while pos < num_bits:
        if not (mask & (1 << pos)):
            pos += 1
            continue
        start = pos
        while pos < num_bits and (mask & (1 << pos)):
            pos += 1
        tree.add(Interval(start, pos, data))
    return tree


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_timeline(
    current_time: int,
    num_processors: int,
    used_pool: IntervalTree,
    future_by_time: dict[int, list[JobEvent]],
    future_times: Iterable[int],
    used_memory_pool: IntervalTree | None = None,
    total_memory: int = 0,
) -> BitmaskTimeline:
    """Build a bitmask timeline from the scheduler's current state.

    This walks forward through future event times, applying JOB_FINISH (clear
    bits) and JOB_START (set bits) to track the exact processor occupancy at
    each segment boundary.

    Parameters
    ----------
    current_time:
        The scheduler's ``current_time``.
    num_processors:
        Total processor count (``scheduler.number_of_processors``).
    used_pool:
        The cluster's ``processors.used_pool`` IntervalTree (current occupancy).
    future_by_time:
        Future events grouped by time (from ``EventQueue.future_events_snapshot()``).
    future_times:
        Sorted list of distinct future event times
        (from ``EventQueue.future_events_snapshot()``).
    used_memory_pool:
        Optional IntervalTree representing current memory occupancy. When
        provided, per-segment ``used_memory`` scalars are computed.
    total_memory:
        Total available memory on the cluster. Used only when
        ``used_memory_pool`` is provided.
    """
    all_times: list[int] = []
    seen: set[int] = set()

    if current_time not in seen:
        all_times.append(current_time)
        seen.add(current_time)

    for t in future_times:
        if t not in seen:
            all_times.append(t)
            seen.add(t)

    all_times.sort()

    occ = _intervaltree_to_bitmask(used_pool)

    track_memory = used_memory_pool is not None
    current_used_memory: int = (
        sum(iv.end - iv.begin for iv in used_memory_pool) if track_memory else 0
    )

    times: list[int] = []
    occupied: list[int] = []
    used_memory_list: list[int] = []

    for t in all_times:
        events = future_by_time.get(t)
        if events is not None:
            for ev in events:
                if ev.type == EventType.JOB_FINISH:
                    job_mask = _job_processor_mask(ev.job)
                    occ &= ~job_mask
                    if track_memory:
                        current_used_memory -= ev.job.requested_memory
                elif ev.type == EventType.JOB_START:
                    job_mask = _job_processor_mask(ev.job)
                    occ |= job_mask
                    if track_memory:
                        current_used_memory += ev.job.requested_memory
        times.append(t)
        occupied.append(occ)
        used_memory_list.append(current_used_memory)

    if not times:
        times.append(current_time)
        occupied.append(occ)
        used_memory_list.append(current_used_memory)

    return BitmaskTimeline(
        times, occupied, num_processors, used_memory_list, total_memory
    )


def _job_processor_mask(job: Job) -> int:
    """Extract the processor bitmask from a job's assigned resources."""
    return _intervaltree_to_bitmask(job.resources.processors)


# ---------------------------------------------------------------------------
# Earliest-Fit Query
# ---------------------------------------------------------------------------


def find_earliest_fit(
    timeline: BitmaskTimeline,
    job: Job,
    ignore_memory: bool = True,
) -> tuple[int, int]:
    """Find the earliest time T where ``job`` fits.

    Returns ``(start_time, allocated_processor_bitmask)``.

    Raises ``AssertionError`` if no feasible time exists.
    """
    requested = job.requested_processors
    duration = job.requested_time
    full = timeline.full_mask
    n = len(timeline.times)
    check_memory = not ignore_memory and job.requested_memory > 0

    for i in range(n):
        t = timeline.times[i]
        window_end = t + duration

        common_free = full
        min_free_memory = timeline.total_memory
        k = i
        while k < n and timeline.times[k] < window_end:
            free_at_k = ~timeline.occupied[k] & full
            common_free &= free_at_k
            if common_free.bit_count() < requested:
                break
            if check_memory:
                min_free_memory = min(
                    min_free_memory, timeline.total_memory - timeline.used_memory[k]
                )
            k += 1

        if common_free.bit_count() >= requested and (
            not check_memory or min_free_memory >= job.requested_memory
        ):
            allocated = select_lowest_bits(common_free, requested)
            return (t, allocated)

    raise AssertionError("No feasible time found for job in bitmask timeline")


# ---------------------------------------------------------------------------
# First-Fit Allocation (lowest bits)
# ---------------------------------------------------------------------------


def select_lowest_bits(mask: int, count: int) -> int:
    """Select the lowest ``count`` set bits from ``mask``."""
    result = 0
    remaining = count
    m = mask
    while remaining > 0 and m:
        lowest = m & (-m)  # isolate lowest set bit
        result |= lowest
        m &= m - 1  # clear lowest set bit
        remaining -= 1
    return result


# ---------------------------------------------------------------------------
# Timeline Update After Scheduling
# ---------------------------------------------------------------------------


def ensure_boundary(timeline: BitmaskTimeline, time: int) -> None:
    """Insert a new segment boundary at ``time`` if not already present.

    The new segment inherits the occupancy mask of the segment it splits.
    """
    idx = bisect_left(timeline.times, time)
    if idx < len(timeline.times) and timeline.times[idx] == time:
        return

    inherited_occ = timeline.occupied[idx - 1] if idx > 0 else 0
    inherited_mem = timeline.used_memory[idx - 1] if idx > 0 else 0
    timeline.times.insert(idx, time)
    timeline.occupied.insert(idx, inherited_occ)
    timeline.used_memory.insert(idx, inherited_mem)


def update_timeline(
    timeline: BitmaskTimeline,
    job: Job,
    start_time: int,
    allocated_mask: int,
    memory_amount: int = 0,
) -> None:
    """Mark processors as occupied in ``[start_time, start_time + execution_time)``.

    CRITICAL: Uses ``job.execution_time`` (actual occupation), NOT
    ``job.requested_time``. This matches ``_add_job_events`` at scheduler.py
    line 196.
    """
    finish_time = start_time + job.execution_time

    ensure_boundary(timeline, start_time)
    ensure_boundary(timeline, finish_time)

    for i in range(len(timeline.times)):
        if timeline.times[i] >= finish_time:
            break
        if timeline.times[i] >= start_time:
            timeline.occupied[i] |= allocated_mask
            if memory_amount > 0:
                timeline.used_memory[i] += memory_amount


# ---------------------------------------------------------------------------
# Bitmask → Resource (for assign_schedule compatibility)
# ---------------------------------------------------------------------------


def _carve_memory_intervals(
    memory_amount: int,
    total_memory: int,
    memory_used_pool: IntervalTree,
    job_id: int | None,
) -> IntervalTree:
    """Carve memory intervals using lowest-address-first algorithm.

    Replicates ``ResourcePool.find()`` exactly.  IntervalTree iteration order
    is not guaranteed, so free intervals are sorted by ``begin`` before the
    greedy allocation pass.
    """
    free: IntervalTree = IntervalTree([Interval(0, total_memory, job_id)])
    for iv in memory_used_pool:
        free.chop(iv.begin, iv.end)

    used: IntervalTree = IntervalTree()
    used_size: int = 0
    for iv in sorted(free, key=lambda x: x.begin):
        temp_size = (iv.end - iv.begin) + used_size
        if temp_size == memory_amount:
            used.add(iv)
            break
        elif temp_size < memory_amount:
            used.add(iv)
            used_size = temp_size
        else:
            used.add(Interval(iv.begin, iv.begin + memory_amount - used_size, job_id))
            break
    return used


def bitmask_to_resource(
    mask: int,
    job_id: int | None = None,
    memory_amount: int = 0,
    total_memory: int = 0,
    memory_used_pool: IntervalTree | None = None,
) -> Resource:
    """Convert an allocated processor bitmask to a ``Resource`` object.

    When *memory_amount* > 0 and *memory_used_pool* is provided, memory
    intervals are carved using the lowest-address-first algorithm and the
    returned ``Resource`` has ``ignore_memory=False``.  Otherwise the
    returned ``Resource`` has ``ignore_memory=True``.
    """
    processors = bitmask_to_intervals(mask, data=job_id)
    if memory_amount > 0 and memory_used_pool is not None:
        memory = _carve_memory_intervals(
            memory_amount, total_memory, memory_used_pool, job_id
        )
        return Resource(processors=processors, memory=memory, ignore_memory=False)
    return Resource(processors=processors, ignore_memory=True)
