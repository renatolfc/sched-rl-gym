#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Callable, Tuple, Generator
from abc import ABC, abstractmethod

from . import job


class Scheduler(ABC):
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
        return filter_update_queue(
            lambda j: j.submission_time <= self.current_time and j.status == job.JobStatus.SCHEDULED,
            self.start_running,
            self.queue_waiting
        )

    def new_completed_jobs(self):
        return filter_update_queue(
            lambda j: j.start_time + j.execution_time >= self.current_time,
            self.complete_job,
            self.queue_running
        )

    def update_queues(self):
        started_running = []
        for j in self.queue_waiting:
            if j.status == job.JobStatus.RUNNING:
                started_running.append(j)
            elif j.status == job.JobStatus.SCHEDULED and self.can_start(j):
                self.start_running(j)
                started_running.append(j)
        for j in started_running:
            self.queue_waiting.remove(j)
        self.queue_running += started_running

        finished_running = []
        for j in self.queue_running:
            if (j.start_time + j.execution_time) <= self.current_time:
                self.complete_job(j)
                finished_running.append(j)
        for j in finished_running:
            self.queue_running.remove(j)
        self.queue_completed += finished_running

    @property
    def free_resources(self):
        return self.number_of_processors - self.used_processors, self.total_memory - self.used_memory

    @abstractmethod
    def schedule(self):
        "Schedules tasks."

    def step(self, time_steps: int = 1):
        for i in range(time_steps):
            self.increase_time(1)
            self.update_queues()
            self.schedule()

    def can_start(self, j: job.Job):
        return self.current_time >= j.submission_time and \
               self.used_processors + j.processors_allocated < self.number_of_processors \
               and self.used_memory + j.memory_use < self.total_memory

    def submit(self, j: job.Job):
        j.submission_time = self.current_time
        j.status = job.JobStatus.SUBMITTED
        self.queue_waiting.append(j)


def filter_update_queue(predicate: Callable, update: Callable, queue: List[job.Job]) -> Tuple[Generator, Generator]:
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
