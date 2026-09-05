"""heap - A Priority Queue based on the `heapq` module."""

try:
    from ._schedgym_rs import Heap
except ImportError:
    import heapq
    import itertools
    from collections.abc import Generator, Iterator
    from typing import Generic, TypeVar, cast

    T = TypeVar("T")
    ENTRY_T = tuple[int, int, list[T | None]]

    class Heap(Generic[T]):
        entry_finder: dict[T | None, ENTRY_T]
        priority_queue: list[ENTRY_T]

        def __init__(self):
            self.priority_queue = []
            self.entry_finder = {}
            self.counter = itertools.count()

        def add(self, item, priority=0) -> None:
            if item in self.entry_finder:
                self.remove(item)
            count = next(self.counter)
            entry = (priority, count, [item])
            self.entry_finder[item] = entry
            heapq.heappush(self.priority_queue, entry)

        def remove(self, item) -> None:
            entry = self.entry_finder.pop(item)
            entry[-1][0] = None

        def pop(self) -> T:
            while self.priority_queue:
                _, _, (item,) = heapq.heappop(self.priority_queue)
                if item is not None:
                    del self.entry_finder[item]  # type: ignore
                    return cast(T, item)
            raise KeyError("pop from an empty priority queue")

        def __iter__(self) -> Iterator[T]:
            return iter(self.heapsort())

        def __contains__(self, item):
            return item in self.entry_finder

        def __len__(self):
            return len(self.entry_finder)

        @property
        def first(self) -> T | None:
            while self.priority_queue and self.priority_queue[0][-1][0] is None:
                heapq.heappop(self.priority_queue)
            if not self.priority_queue:
                return None
            return cast(T, self.priority_queue[0][-1][0])

        def heapsort(self) -> Generator[T, None, None]:
            h = list(self.priority_queue)
            while h:
                entry = heapq.heappop(h)[-1][0]
                if entry is not None:
                    yield cast(T, entry)
