"""pool - Resource Pool management (see :class:`schedgym.cluster.Cluster`)."""

import copy
import enum
from collections.abc import Iterable

try:
    from ._schedgym_rs import (
        Interval,
        IntervalTree,
    )
    from ._schedgym_rs import (
        ResourcePool as _RustResourcePool,
    )

    _USE_RUST = True
except ImportError:
    from intervaltree import Interval, IntervalTree

    _USE_RUST = False


class ResourceType(enum.IntEnum):
    """Enumeration to determine which kind of resource we're managing."""

    CPU = 1
    MEMORY = 0


if _USE_RUST:
    ResourcePool = _RustResourcePool
else:

    class ResourcePool:
        used_pool: IntervalTree

        def __init__(
            self,
            resource_type: ResourceType,
            size: int,
            used_pool: IntervalTree | None = None,
        ):
            self.size = size
            self.used_resources = 0
            self.type = resource_type
            if used_pool is None:
                self.used_pool = IntervalTree()
            else:
                self.used_pool = used_pool
                self.used_resources = sum(ResourcePool.measure(i) for i in used_pool)

        def clone(self):
            return copy.deepcopy(self)

        @property
        def free_resources(self) -> int:
            return self.size - self.used_resources

        def fits(self, size) -> bool:
            if size <= 0:
                raise AssertionError("Can't allocate zero resources")
            return size <= self.free_resources

        @staticmethod
        def measure(interval: Interval):
            return interval.end - interval.begin

        def find(self, size: int, data: int | None = None) -> IntervalTree:
            used = IntervalTree()
            if not self.fits(size):
                return used
            free = IntervalTree([Interval(0, self.size, data)])
            used_size: int = 0
            for interval in self.used_pool:
                free.chop(interval.begin, interval.end)
            for interval in free:
                temp_size = ResourcePool.measure(interval) + used_size
                if temp_size == size:
                    used.add(interval)
                    break
                if temp_size < size:
                    used.add(interval)
                    used_size = temp_size
                else:
                    used.add(
                        Interval(
                            interval.begin,
                            interval.begin + size - used_size,
                            data,
                        )
                    )
                    break
            return used

        def allocate(self, intervals: Iterable[Interval]) -> None:
            for i in intervals:
                if self.used_resources + self.measure(i) > self.size:
                    raise AssertionError("Tried to allocate past size of resource pool")
                self.used_pool.add(i)
                self.used_resources += self.measure(i)

        def free(self, intervals: Iterable[Interval]) -> None:
            for i in intervals:
                if i not in self.used_pool:
                    raise AssertionError("Tried to free unused resource set")
                self.used_pool.remove(i)
                self.used_resources -= self.measure(i)

        @property
        def intervals(self) -> list[Interval]:
            return list(self.used_pool)

        def __repr__(self):
            return (
                f"ResourcePool(resource_type={self.type}, "
                f"size={self.size}, used_pool={self.used_pool})"
            )
