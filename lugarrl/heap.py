#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq, itertools
from typing import Generic, TypeVar, List, Dict, Generator, Optional, Iterator

T = TypeVar('T')
REMOVED: str = '<removed-entry>'


class Heap(Generic[T]):
    priority_queue: List[List]
    entry_finder: Dict[T, List]
    auto_vacuum: int

    def __init__(self, auto_vacuum=1000):
        self.priority_queue = []
        self.entry_finder = {}
        self.counter = itertools.count()
        self.auto_vacuum = auto_vacuum

    def add(self, item, priority=0) -> None:
        'Add a new item or update the priority of an existing item'
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.priority_queue, entry)
        if (count % self.auto_vacuum) == 0:
            self.vacuum()

    def remove(self, item) -> None:
        'Mark an existing item as removed. Raise KeyError if not found.'
        entry = self.entry_finder.pop(item)
        entry[-1] = REMOVED

    def pop(self) -> T:
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.priority_queue:
            priority, count, item = heapq.heappop(self.priority_queue)
            if item is not REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def vacuum(self) -> None:
        tmp: List = []
        for (priority, count, item) in self.priority_queue:
            if item is not REMOVED:
                heapq.heappush(tmp, [priority, count, item])
        self.priority_queue = tmp

    def __iter__(self) -> Iterator[T]:
        return iter(self.heapsort())

    def __contains__(self, item):
        return item in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)

    @property
    def first(self) -> Optional[T]:
        if len(self.entry_finder) == 0:
            return None
        for (priority, count, item) in self.priority_queue:
            if item is not REMOVED:
                return item
        return None

    def heapsort(self) -> Generator[T, None, None]:
        h = [e for e in self.priority_queue]
        while h:
            entry = heapq.heappop(h)[-1]
            if entry is not REMOVED:
                yield entry
