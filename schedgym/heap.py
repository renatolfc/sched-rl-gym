#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""heap - A Priority Queue based on the `heapq` module."""

import heapq
import itertools
from typing import Generic, TypeVar, List, Dict, Generator, Optional, Iterator
from typing import Tuple, cast

T = TypeVar('T')
ENTRY_T = Tuple[int, int, List[Optional[T]]]


class Heap(Generic[T]):
    """A Priority Queue that is backed by a heap data structure.

    To reduce the computational cost of key removal, this class wastes a bit
    memory by *not* actually deleting items.
    """

    entry_finder: Dict[Optional[T], ENTRY_T]
    'Cache to check in O(1) whether an entry exists in the heap.'
    priority_queue: List[ENTRY_T]
    'The actual priority queue, implemented as a list with heap ordering.'

    def __init__(self):
        """Initializes the heap.

        """
        self.priority_queue = []
        self.entry_finder = {}
        self.counter = itertools.count()

    def add(self, item, priority=0) -> None:
        """Add a new item or update the priority of an existing item"""
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = (priority, count, [item])
        self.entry_finder[item] = entry
        heapq.heappush(self.priority_queue, entry)

    def remove(self, item) -> None:
        """Mark an existing item as removed. Raise KeyError if not found."""
        entry = self.entry_finder.pop(item)
        entry[-1][0] = None

    def pop(self) -> T:
        """Remove and return the lowest priority task.

        Raises KeyError if empty."""
        while self.priority_queue:
            _, _, (item,) = heapq.heappop(self.priority_queue)
            if item is not None:
                del self.entry_finder[item]  # type: ignore
                return cast(T, item)
        raise KeyError('pop from an empty priority queue')

    def __iter__(self) -> Iterator[T]:
        return iter(self.heapsort())

    def __contains__(self, item):
        return item in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)

    @property
    def first(self) -> Optional[T]:
        """Returns the "first" item (highest priority item) in the Heap."""
        if len(self.entry_finder) == 0:
            return None
        for (_, _, (item,)) in self.priority_queue:
            if item is not None:
                return cast(T, item)
        return None

    def heapsort(self) -> Generator[T, None, None]:
        """Generator that iterates over all elements in the heap in priority
        order."""
        h = [e for e in self.priority_queue]
        while h:
            entry = heapq.heappop(h)[-1][0]
            if entry is not None:
                yield cast(T, entry)
