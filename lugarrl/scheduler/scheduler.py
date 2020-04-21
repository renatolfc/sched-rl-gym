#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Iterable, Tuple, Dict, Any, Sequence, Union

import collections
import numpy as np

from lugarrl.cluster import Cluster
from lugarrl.job import Job, JobStatus, Resource
from lugarrl.event import JobEvent, EventType, EventQueue


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
    cluster: Cluster
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
        self.cluster = Cluster(number_of_processors, total_memory)

    @property
    def all_jobs(self) -> List[Job]:
        return self.queue_completed + self.queue_running + self.queue_waiting \
            + self.queue_admission

    @property
    def jobs_in_system(self) -> List[Job]:
        return self.queue_running + self.queue_waiting + self.queue_admission

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
        if not job.resources or not job.proper:
            raise AssertionError(
                "Malformed job submitted either with no processors, "
                "or with insufficient number of "
                "processors"
            )
        start = JobEvent(
            time, EventType.JOB_START, job
        )
        finish = start.clone()
        finish.time += job.execution_time
        finish.type = EventType.JOB_FINISH
        self.job_events.add(start)
        self.job_events.add(finish)

    @property
    def free_resources(self) -> Tuple[int, int]:
        return self.number_of_processors - self.used_processors, self.total_memory - self.used_memory

    def step(self, offset: int = None) -> bool:
        if offset is None:
            offset = 1
        if offset < 0:
            raise AssertionError("Tried to move backwards in time")

        scheduled = False
        for _ in range(offset):
            if self.queue_admission:
                scheduled = True
                self.schedule()
            present = self.job_events.step(1)
            self.cluster = self.play_events(present, self.cluster, update_queues=True)
            self.current_time += 1
        return scheduled

    def play_events(self, events: Iterable[JobEvent], cluster: Cluster,
                    update_queues: bool = False) -> Cluster:
        for event in events:
            if event.type == EventType.JOB_START:
                cluster.allocate(event.job)
                if update_queues:
                    self.start_running(event.job)
            elif event.type == EventType.JOB_FINISH:
                cluster.free(event.job)
                if update_queues:
                    self.complete_job(event.job)
            else:
                raise RuntimeError("Unexpected event type found")
        return cluster

    @staticmethod
    def fits(time: int, job: Job, cluster: Cluster, events: Iterable[JobEvent]) \
            -> Resource:
        return cluster.find_resources_at_time(time, job, events)

    def can_schedule_now(self, job: Job) -> Resource:
        cluster = self.cluster.clone()
        for event in (e for e in self.job_events if e.time <= self.current_time):
            if event.type == EventType.JOB_START:
                cluster.allocate(event.job)
            elif event.type == EventType.JOB_FINISH:
                cluster.free(event.job)
        return cluster.find_resources_at_time(
            self.current_time, job, self.job_events
        )

    def find_first_time_for(self, job: Job) -> Tuple[int, Resource]:
        if (not self.job_events.next) or self.job_events.next.time > self.current_time:
            resources = self.cluster.find_resources_at_time(self.current_time, job, self.job_events)
            if resources:
                return self.current_time, resources

        near_future: Dict[int, List[JobEvent]] = defaultdict(list)
        for e in self.job_events:
            near_future[e.time].append(e)

        cluster = self.cluster.clone()
        for time in sorted(near_future):
            cluster = self.play_events(near_future[time], cluster)
            resources = cluster.find_resources_at_time(time, job, self.job_events)
            if resources:
                return time, resources

        raise AssertionError('Failed to find time for job, even in the far future.')

    def submit(self, job: Union[Job, Sequence[Job]]) -> None:
        if isinstance(job, collections.Iterable):
            for j in job:
                self._submit(j)
        else:
            self._submit(job)

    def _submit(self, job: Job) -> None:
        if job.requested_processors > self.number_of_processors:
            raise RuntimeError(
                'Impossible to allocate resources for job bigger than cluster.'
            )
        job.submission_time = self.current_time
        job.status = JobStatus.SUBMITTED
        self.queue_admission.append(job)

    def state(self, timesteps: int, job_slots: int, backlog_size: int):
        near_future: Dict[int, List[JobEvent]] = defaultdict(list)
        for e in (e for e in self.job_events
                  if e.time < self.current_time + timesteps + 1):
            near_future[e.time - self.current_time].append(e)

        memory = np.zeros((timesteps, self.total_memory))
        processors = np.zeros((timesteps, self.number_of_processors))
        cluster = self.cluster.clone()
        for t in range(timesteps):
            if t in near_future:
                cluster = self.play_events(near_future[t], cluster)
            processors[t, :], memory[t, :] = cluster.state
        state = (processors, memory)

        positions = {}
        memory = np.zeros((job_slots, timesteps, self.total_memory))
        processors = np.zeros((job_slots, timesteps, self.number_of_processors))
        for i, job in enumerate(self.queue_admission):
            if i == job_slots:
                break
            if job.slot_position is None:
                if i in positions:
                    empty = set(range(job_slots)) - set(list(positions.keys()))
                    positions[list(empty)[0]] = job
                else:
                    positions[i] = job
            else:
                if job.slot_position in positions:
                    empty = set(range(job_slots)) - set(list(positions.keys()))
                    tmp = positions[job.slot_position]
                    positions[job.slot_position] = job
                    positions[list(empty)[0]] = tmp
                else:
                    positions[job.slot_position] = job
        for i, job in positions.items():
            job.slot_position = i
            processors[i, :, :], memory[i, :, :] = cluster.get_job_state(job, timesteps)
        jobs = (processors, memory)

        backlog = np.zeros((backlog_size,))
        backlog[:min(max(len(self.queue_admission) - job_slots, 0), backlog_size)] = 1.0

        return state, jobs, backlog

    def assign_schedule(self, job, resources, time):
        job.status = JobStatus.WAITING
        job.resources.memory = resources.memory
        job.resources.processors = resources.processors
        job.start_time = time
        self.add_job_events(job, time)
        self.queue_waiting.append(job)

    @abstractmethod
    def schedule(self) -> Any:
        "Schedules tasks."
