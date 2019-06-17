#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple, Dict

from .job import Job, JobStatus
from .event import JobEvent, EventType, EventQueue
from .resource_pool import ResourceType, ResourcePool, Interval, IntervalTree


class Scheduler(ABC):
    used_memory: int
    current_time: int
    total_memory: int
    used_processors: int
    number_of_processors: int
    queue_waiting: List[Job]
    queue_running: List[Job]
    queue_admission: List[Job]
    queue_completed: List[Job]
    processor_pool: ResourcePool
    job_events: EventQueue[JobEvent]

    def __init__(self, number_of_processors, total_memory):
        self.number_of_processors = number_of_processors
        self.total_memory = total_memory

        self.queue_waiting = []
        self.queue_running = []
        self.queue_completed = []
        self.queue_admission = []

        self.used_memory = 0
        self.current_time = 0
        self.used_processors = 0
        self.job_events = EventQueue(self.current_time - 1)
        self.processor_pool = ResourcePool(ResourceType.CPU, number_of_processors)

    @property
    def all_jobs(self) -> List[Job]:
        return self.queue_completed + self.queue_running + self.queue_waiting + self.queue_admission

    @property
    def makespan(self) -> int:
        return max([j.finish_time for j in self.queue_completed])

    def start_running(self, j: Job) -> None:
        self.queue_waiting.remove(j)
        self.queue_running.append(j)

        j.status = JobStatus.RUNNING
        self.used_memory += j.memory_use
        self.used_processors += j.processors_allocated
        j.wait_time = j.start_time - j.submission_time

    def complete_job(self, j: Job) -> None:
        self.queue_running.remove(j)
        self.queue_completed.append(j)

        j.status = JobStatus.COMPLETED
        j.finish_time = j.start_time + j.execution_time
        self.used_memory -= j.memory_use
        self.used_processors -= j.processors_allocated

    def add_job_events(self, job: Job, time: int) -> None:
        if not job.resources_used or not job.proper:
            raise AssertionError(
                "Malformed job submitted either with no processors, or with insufficient number of "
                "processors"
            )
        start = JobEvent(
            time, EventType.JOB_START, job
        )
        finish = start.clone()
        finish.time += job.execution_time
        finish.event_type = EventType.JOB_FINISH
        self.job_events.add(start)
        self.job_events.add(finish)

    @property
    def free_resources(self) -> Tuple[int, int]:
        return self.number_of_processors - self.used_processors, self.total_memory - self.used_memory

    def step(self, offset: int = 1) -> None:
        if offset < 0:
            raise AssertionError("Tried to move backwards in time")

        if self.queue_admission:
            self.schedule()

        present = self.job_events.step(offset)
        self.play_events(present, self.processor_pool, update_queues=True)
        self.current_time += offset

        self.schedule()

    def play_events(self, events: Iterable[JobEvent], pool: ResourcePool,
                    update_queues: bool = False) -> ResourcePool:
        for event in events:
            if event.event_type == EventType.JOB_START:
                pool.allocate(event.processors)
                if update_queues:
                    self.start_running(event.job)
            elif event.event_type == EventType.JOB_FINISH:
                pool.deallocate(event.processors)
                if update_queues:
                    self.complete_job(event.job)
            else:
                raise RuntimeError("Unexpected event type found")
        return pool

    @staticmethod
    def fits(time: int, job: Job, pool: ResourcePool, events: Iterable[JobEvent]) \
            -> Iterable[Interval]:
        """Checks whether a job fits the system starting at a given time.

        It is required that the pool is updated up to :param time:.
        """
        processors_touched = IntervalTree(pool.used_pool)
        for event in (e for e in events if time + 1 <= e.time < job.requested_time + time):
            if event.event_type == EventType.JOB_START:
                for i in event.processors:
                    processors_touched.add(i)
        processors_touched.merge_overlaps()
        current_pool = ResourcePool(pool.type, pool.size, processors_touched)
        return current_pool.find(job.requested_processors)

    def find_first_time_for(self, job: Job) -> Tuple[int, Iterable[Interval]]:
        if (not self.job_events.next) or self.job_events.next.time > self.current_time:
            resources = self.fits(self.current_time, job, self.processor_pool, self.job_events)
            if resources:
                return self.current_time, resources

        near_future: Dict[int, List[JobEvent]] = defaultdict(list)
        for e in self.job_events:
            near_future[e.time].append(e)

        resource_pool = self.processor_pool.clone()
        for time in sorted(near_future):
            resource_pool = self.play_events(near_future[time], resource_pool)
            resources = self.fits(time, job, resource_pool, self.job_events)
            if resources:
                return time, resources

        raise AssertionError('Failed to find time for job, even in the far future.')

    def submit(self, job: Job) -> None:
        if job.requested_processors > self.number_of_processors:
            raise RuntimeError(
                'Impossible to allocate resources for job bigger than cluster.'
            )
        job.submission_time = self.current_time
        job.status = JobStatus.SUBMITTED
        self.queue_admission.append(job)

    @abstractmethod
    def schedule(self) -> None:
        "Schedules tasks."
