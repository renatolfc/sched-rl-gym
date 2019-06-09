#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq, itertools

REMOVED: str = '<removed-entry>'


class Heap(object):
    def __init__(self, autovacuum=1000):
        self.priority_queue = []
        self.entry_finder = {}
        self.counter = itertools.count()
        self.autovacuum = autovacuum

    def add(self, item, priority=0):
        'Add a new item or update the priority of an existing item'
        if item in self.entry_finder:
            self.remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.priority_queue, entry)
        if (count % self.autovacuum) == 0:
            self.vacuum()

    def remove(self, item):
        'Mark an existing item as removed. Raise KeyError if not found.'
        entry = self.entry_finder.pop(item)
        entry[-1] = REMOVED

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.priority_queue:
            priority, count, item = heapq.heappop(self.priority_queue)
            if item is not REMOVED:
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def vacuum(self):
        tmp = []
        for (priority, count, item) in self.priority_queue:
            if item is not REMOVED:
                heapq.heappush(tmp, [priority, count, item])
        self.priority_queue = tmp

    def __iter__(self):
        return iter(self.entry_finder)

    def __contains__(self, item):
        return item in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)

    @property
    def first(self):
        if len(self.entry_finder) == 0:
            return None
        for (priority, count, item) in self.priority_queue:
            if item is not REMOVED:
                return item

    def heapsort(self):
        h = []
        for entry in self.priority_queue:
            if entry[-1] is not REMOVED:
                heapq.heappush(h, entry)

        return (heapq.heappop(h)[-1] for _ in range(len(h)))

