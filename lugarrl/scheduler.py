#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Callable, Tuple, Generator
from abc import ABC, abstractmethod

from . import job


class Scheduler(object):
    queue_waiting: List[job.Job]
    queue_running: List[job.Job]
    queue_completed: List[job.Job]

    def __init__(self, number_of_processors, total_memory):
        self.number_of_processors = number_of_processors
        self.total_memory = total_memory
        self.queue_waiting, self.queue_running, self.queue_completed = [], [], []

        self.used_memory = 0
        self.current_time = 0
        self.used_processors = 0
        self.last_completed = []

    @property
    def all_jobs(self):
        return self.queue_completed + self.queue_running + self.queue_waiting

    def increase_time(self, offset=1):
        if offset < 0:
            raise AssertionError("Tried to decrement time.")
        self.current_time += offset

    def makespan(self):
        return max([j.finish_time for j in self.all_jobs])

    def start_running(self, j: job.Job):
        j.status = job.JobStatus.RUNNING
        j.start_time = self.current_time
        self.used_memory += j.memory_use
        self.used_processors += j.processors_allocated
        j.wait_time = j.start_time - j.submission_time

    def complete_job(self, j: job.Job):
        j.status = job.JobStatus.COMPLETED
        j.finish_time = self.current_time
        self.used_memory -= j.memory_use
        self.used_processors -= j.processors_allocated

    def new_running_jobs(self):
        return _filter_update_queue(
            lambda j: j.submission_time <= self.current_time and j.status == job.JobStatus.SCHEDULED,
            self.start_running,
            self.queue_waiting
        )

    def new_completed_jobs(self):
        return _filter_update_queue(
            lambda j: j.start_time + j.execution_time >= self.current_time,
            self.complete_job,
            self.queue_running
        )

    def update_queues(self):
        started_running, still_waiting = self.new_running_jobs()
        completed, still_running = self.new_completed_jobs()

        self.last_completed = list(completed)
        self.queue_completed += self.last_completed
        self.queue_running = list(still_running) + list(started_running)
        self.queue_waiting = list(still_waiting)

    @property
    def free_resources(self):
        return self.number_of_processors - self.used_processors, self.total_memory - self.used_memory

    @abstractmethod
    def schedule(self):
        pass

    def step(self, time_steps: int = 1):
        self.increase_time(time_steps)
        self.update_queues()
        self.schedule()

    def can_schedule(self, j: job.Job):
        return self.used_processors + j.processors_allocated < self.number_of_processors \
               and self.used_memory + j.memory_use < self.total_memory

    def submit(self, j: job.Job):
        j.submission_time = self.current_time
        self.queue_waiting.append(j)


def _filter_update_queue(predicate: Callable, update: Callable, queue: List[job.Job]) -> Tuple[Generator, Generator]:
    true_idx, false_idx = [], []
    for i in range(len(queue)):
        if predicate(queue[i]):
            true_idx.append(i)
        else:
            false_idx.append(i)

    def true_generator():
        for idx in true_idx:
            update(queue[idx])
            yield queue[idx]

    return true_generator(), (queue[i] for i in false_idx)
