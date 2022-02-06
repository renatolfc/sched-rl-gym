#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""scheduler - Module with basic scheduling functionality.

This is the core of the simulator, since this module contains functionality
that interacts with all other components.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    List,
    Iterable,
    Tuple,
    Dict,
    Any,
    Union,
    NamedTuple,
    Optional,
)

import collections.abc

import numpy as np

from schedgym.cluster import Cluster
from schedgym.job import Job, JobStatus, Resource
from schedgym.event import JobEvent, EventType, EventQueue


class Stats(NamedTuple):
    """A named tuple with scheduling statistics"""

    utilization: float
    load: float
    slowdown: float
    makespan: float
    bsld: float


class Scheduler(ABC):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    """Base class for scheduling.

    This class implements the core scheduling primitives common to all
    schedulers, and it also manages the "connection" with Cluster objects
    to manage them.

    Internally, the scheduler manages four general "queues":
        * Admission: For jobs that have been submitted, but to which the
          scheduler hasn't made a decision yet
        * Waiting: For jobs that the scheduler has already generated an
          schedule, but that haven't been started yet
        * Running: For jobs that have started execution, but hasn't
          finished yet
        * Completed: For jobs that have finished execution

    Parameters
    ----------
        number_of_processors : int
            The number of processors in the system
        total_memory : int
            The amount of memory in the system
        ignore_memory : bool
            Whether memory should be ignored when making decisions, or not
    """

    used_memory: int
    current_time: int
    total_memory: int
    used_processors: int
    need_schedule_call: bool
    number_of_processors: int
    queue_waiting: List[Job]
    queue_running: List[Job]
    queue_admission: List[Job]
    queue_completed: List[Job]
    cluster: Cluster
    job_events: EventQueue[JobEvent]
    stats: Dict[int, Stats]

    def __init__(
        self, number_of_processors, total_memory, ignore_memory=False
    ):
        self.number_of_processors = number_of_processors
        self.total_memory = total_memory

        self.queue_waiting = []
        self.queue_running = []
        self.queue_completed = []
        self.queue_admission = []

        self.stats = {}
        self.used_memory = 0
        self.current_time = 0
        self.used_processors = 0
        self.ignore_memory = ignore_memory
        self.job_events = EventQueue(self.current_time - 1)
        self.cluster = Cluster(
            number_of_processors, total_memory, ignore_memory
        )
        self.need_schedule_call = False
        'Tracks whether we might need to schedule jobs'

    @property
    def all_jobs(self) -> List[Job]:
        """Returns a list of all the jobs that ever got into the system"""
        return (
            self.queue_completed
            + self.queue_running
            + self.queue_waiting
            + self.queue_admission
        )

    @property
    def slowdown(self) -> List[float]:
        """Returns the slowdown of all completed jobs"""
        return [j.slowdown for j in self.queue_completed]

    @property
    def jobs_in_system(self) -> List[Job]:
        """Returns a list with all the jobs that haven't completed yet"""
        return self.queue_running + self.queue_waiting + self.queue_admission

    @property
    def makespan(self) -> int:
        """Computes the makespan of all finished jobs"""
        return max([0] + [j.finish_time for j in self.queue_completed])

    @property
    def load(self) -> float:
        """Computes the current load in the system.

        The load is the ratio between the number of requested processors and
        the number of processors in the system.
        """
        requested_processors = sum(
            [j.requested_processors for j in self.jobs_in_system]
        )
        return requested_processors / self.number_of_processors

    @property
    def utilization(self) -> float:
        """Instant processor utilization."""
        return self.used_processors / self.number_of_processors

    @property
    def bounded_slowdown(self) -> List[float]:
        """Computes the bounded slowdown for all completed jobs"""
        return [j.bounded_slowdown for j in self.queue_completed]

    def _start_running(self, j: Job) -> None:
        """Starts running job `j`.

        Parameters
        ----------
            j : Job
                The job to start running
        """
        self.queue_waiting.remove(j)
        self.queue_running.append(j)

        j.status = JobStatus.RUNNING
        self.used_memory += j.memory_use
        self.used_processors += j.processors_allocated
        j.wait_time = j.start_time - j.submission_time

    def _complete_job(self, j: Job) -> None:
        """Marks a job as completed.

        Parameters
        ----------
            j : Job
                The job to mark completed
        """
        self.queue_running.remove(j)
        self.queue_completed.append(j)

        j.status = JobStatus.COMPLETED
        j.finish_time = j.start_time + j.execution_time
        self.used_memory -= j.memory_use
        self.used_processors -= j.processors_allocated

    def _add_job_events(
        self, job: Job, time: int
    ) -> Tuple[JobEvent, JobEvent]:
        """Adds start and finish events for a job to the current events.

        Parameters
        ----------
            job : Job
                The job whose events are to be added to the system
            time : int
                The time step to associate the start event with
        """
        if not job.resources or not job.proper:
            raise AssertionError(
                'Malformed job submitted either with no processors, '
                'or with insufficient number of '
                'processors'
            )
        start = JobEvent(time, EventType.JOB_START, job)
        finish = start.clone()
        finish.time += job.execution_time
        finish.type = EventType.JOB_FINISH
        self.job_events.add(start)
        self.job_events.add(finish)

        return start, finish

    @property
    def free_resources(self) -> Tuple[int, int]:
        """Returns the amount of free resources in the system."""
        return (
            self.number_of_processors - self.used_processors,
            self.total_memory - self.used_memory,
        )

    def step(self, offset: int = None) -> bool:
        """Steps the simulation

        Parameters
        ----------
            offset : int
                The number of time steps to take (must be >= 0)
        """
        if offset is None:
            offset = 1
        if offset < 0:
            raise AssertionError('Tried to move backwards in time')

        scheduled = False
        for _ in range(offset):
            if self.need_schedule_call or (
                self.queue_admission
                and self.job_events.first
                and self.job_events.first.time == self.current_time
            ):
                self.need_schedule_call = False
                scheduled = True
                self.schedule()
            present = self.job_events.step(1)
            self.cluster = self.play_events(
                present, self.cluster, update_queues=True
            )
            self.current_time += 1
        return scheduled

    def play_events(
        self,
        events: Iterable[JobEvent],
        cluster: Cluster,
        update_queues: bool = False,
    ) -> Cluster:
        """Play events from a given event queue, updating state accordingly.

        On top of playing the events, this also updates job statistics,
        which can be queried at any given time.

        After execution, the current state of the cluster is returned.

        This method is used by a number of operations: both to find future
        schedules for jobs and to check whether a job can be added at a given
        time step. For this reason, an optional argument is included to define
        whether to update queues or not.

        Parameters
        ----------
            events : Iterable[JobEvent]
                The events to play
            cluster : Cluster
                The cluster to operate on when playing events
            update_queues : bool
                Whether to update queues when job start and job finished events
                are found.
        """
        for event in events:
            if not event:
                continue
            if event.type == EventType.JOB_START:
                cluster.allocate(event.job)
                if update_queues:
                    self._start_running(event.job)
                    self.update_stats()
            elif event.type == EventType.JOB_FINISH:
                cluster.free(event.job)
                if update_queues:
                    self._complete_job(event.job)
                    self.update_stats()
            else:
                raise RuntimeError('Unexpected event type found')
        return cluster

    @staticmethod
    def fits(
        time: int, job: Job, cluster: Cluster, events: Iterable[JobEvent]
    ) -> Resource:
        """Checks whether a job fits a given cluster at a given time.

        Once again, this requires an iterable of events and a cluster to
        operate on to check whether the job fits the cluster.

        Parameters
        ----------
            job : Job
                The job to check
            cluster : Cluster
                The cluster to operate on
            events : Iterable[JobEvent]
                An iterable that provides the job events this scheduler will
                operate on

        Returns:
            Resource: The set of resources (when found) or an empty set of
            resources (when the job won't fit the cluster).
        """
        return cluster.find_resources_at_time(time, job, events)

    def some_job_fits(self, job_slots: slice = slice(0, None)):
        """Checks whether any jobs in the admission queue fits _right now_."""

        return any(
            [self.cluster.fits(j) for j in self.queue_admission[job_slots]]
        )

    def can_schedule_now(self, job: Job) -> Resource:
        """Checks whether a job can be scheduled in the current cluster now.

        This is a special case of :func:`fits` in which we're operating right
        now with the current cluster.

        Parameters
        ----------
            job : Job
                The job to check.
        """
        cluster = self.cluster.clone()
        events = filter(lambda e: e.time <= self.current_time, self.job_events)
        for event in events:
            if event.type == EventType.JOB_START:
                cluster.allocate(event.job)
            elif event.type == EventType.JOB_FINISH:
                cluster.free(event.job)
        return cluster.find_resources_at_time(
            self.current_time, job, self.job_events
        )

    def find_first_time_for(self, job: Job) -> Tuple[int, Resource]:
        """Finds the first time stamp on which we can start a job.

        Parameters
        ----------
            job : Job
                The job to find a time for
        """

        if (not self.job_events.next) or (
            self.job_events.next.time > self.current_time
        ):
            resources = self.cluster.find_resources_at_time(
                self.current_time, job, self.job_events
            )
            if resources:
                return self.current_time, resources

        near_future: Dict[int, List[JobEvent]] = defaultdict(list)
        for e in self.job_events:
            near_future[e.time].append(e)

        cluster = self.cluster.clone()
        for time in sorted(near_future):
            cluster = self.play_events(near_future[time], cluster)
            resources = cluster.find_resources_at_time(
                time, job, self.job_events
            )
            if resources:
                return time, resources

        raise AssertionError(
            'Failed to find time for job, even in the far future.'
        )

    def submit(self, job: Union[Job, Iterable[Optional[Job]]]) -> None:
        """Submits a new job to the system.

        Parameters
        ----------
            job : Union[Job, Sequence[Job]]
                Can either be a single job, or a sequence of jobs. If
                a sequence, all jobs in the sequence are submitted at the same
                time.
        """
        if isinstance(job, collections.abc.Iterable):
            for j in job:
                self._submit(j)
        else:
            self._submit(job)
        self.need_schedule_call = True

    def _submit(self, job: Optional[Job]) -> None:
        """Internal implementation of job submission.

        Adds the new job to the `submission_queue` and sets job status to
        `JobStatus.SUBMITTED`.
        """
        if job is None:
            return

        if job.requested_processors > self.number_of_processors:
            raise RuntimeError(
                'Impossible to allocate resources for job bigger than cluster.'
            )
        job.submission_time = self.current_time
        job.status = JobStatus.SUBMITTED

        # Compute statistics to be used in state representation {{{
        job.queue_size = len(self.queue_admission)
        job.queued_work = sum(
            [
                j.requested_time * j.requested_processors
                for j in self.queue_admission
            ]
        )
        job.free_processors = self.cluster.state[0][0]
        # }}}

        self.queue_admission.append(job)

    def state(self, timesteps: int, job_slots: int, smdp: bool = False):
        """Returns the current state of the cluster as viewed by the scheduler.

        The state representation used here is deeply inspired by the DeepRM
        state representation, meaning it will return three blocks of
        information:
        * The current status of processors and memory used in the system
        * A select number of jobs in the admission queue
        * A "backlog" representing the presence or absence of jobs in the queue
        (for jobs that didn't make into the previous representation)

        Parameters
        ----------
            timesteps : int
                The number of time steps to look into the future
            job_slots : int
                The number of job slots to use (the amount of jobs in the
                admission queue to represent)
            smdp : bool
                Whether this is an SMDP and steps should be based on events,
                not time
        """
        # Gets all events between now and `timesteps` {{{
        near_future: Dict[int, List[JobEvent]] = defaultdict(list)
        if smdp:
            last_time = 0
            for e in self.job_events:
                if e.time < self.current_time:
                    continue
                last_time = e.time - self.current_time
                near_future[last_time].append(e)
                if len(near_future) > timesteps:
                    break
            if len(near_future) < timesteps:
                for i in range(last_time + 1, last_time + 1 + timesteps - len(near_future)):
                    near_future[last_time + i].append([])  # type: ignore
            elif len(near_future) > timesteps:
                near_future = {
                    k: v for i, (k, v) in enumerate(near_future.items()) if i < timesteps
                }
        else:
            for e in filter(
                lambda e: e.time < self.current_time + timesteps + 1,
                self.job_events
            ):
                near_future[e.time - self.current_time].append(e)
        # }}}

        # Gets the state representation of currently in use resources {{{
        tmp = []
        cluster = self.cluster.clone()
        for t in (near_future.keys() if smdp else range(timesteps)):
            if t in near_future:
                cluster = self.play_events(near_future[t], cluster)
            tmp.append((t, *cluster.state))
        state = list(zip(*tmp))
        if self.ignore_memory and not cluster.ignore_memory:
            state = state[:-1]
        # }}}

        # Gets the representation of jobs in `job_slots` {{{
        jobs = [
            j.state
            for i, j in enumerate(self.queue_admission)
            if i < job_slots
        ]
        for i, job in enumerate(self.queue_admission):
            if i >= job_slots:
                break
            job.slot_position = i
        jobs += [Job().state for _ in range(job_slots - len(jobs))]
        # }}}

        # Gets the backlog {{{
        backlog = max(len(self.queue_admission) - len(jobs), 0)
        # }}}

        return state, jobs, backlog

    def assign_schedule(
        self, job, resources, time
    ) -> Tuple[JobEvent, JobEvent]:
        """Assigns a schedule to a job.

        What this means is that the job is removed from the admission queue
        and is put into the "waiting" queue, which contains jobs that *will*
        run and already have an schedule. Also changes job status and assigns
        resources to a joub, along with the time it will start running.

        Parameters
        ----------
            job : Job
                The job to be assigned a schedule
            resources : Resource
                The set of resources the job will use
            time : int
                The start time of the job
        """
        job.status = JobStatus.WAITING
        job.resources.memory = resources.memory
        job.resources.processors = resources.processors
        job.resources.ignore_memory = resources.ignore_memory
        job.start_time = time
        self.queue_waiting.append(job)
        return self._add_job_events(job, time)

    @abstractmethod
    def schedule(self) -> Any:
        """Schedules tasks."""

    def update_stats(self) -> None:
        """Updates the usage statistics of the system.

        Statistics are only computed when job events happen in the cluster.
        """
        self.stats[self.current_time] = Stats(
            self.utilization,
            self.load,
            np.mean(self.slowdown) if self.queue_completed else 0.0,
            self.makespan,
            np.mean(self.bounded_slowdown) if self.queue_completed else 0.0,
        )
