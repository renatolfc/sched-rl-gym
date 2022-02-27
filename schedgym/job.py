#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""job - Classes for jobs in the simulator.
"""

import enum

import random
import warnings

from collections import namedtuple

from .resource import Resource, PrimaryResource

JobState = namedtuple(
    'JobState',
    [
        'submission_time',
        'requested_time',
        'requested_memory',
        'requested_processors',
        'queue_size',
        'queued_work',
        'free_processors',
        'can_schedule_now',
    ],
)


class JobStatus(enum.IntEnum):
    """An enumeration for different states of a job within our simulator."""

    SUBMITTED = 0
    RUNNING = 1
    WAITING = 2
    COMPLETED = 3
    SCHEDULED = 4


class SwfJobStatus(enum.IntEnum):
    """An enumeration for different states of a job in the SWF_.

    .. _SWF: https://www.cs.huji.ac.il/labs/parallel/workload/swf.html
    """

    FAILED = 0
    COMPLETED = 1
    PARTIAL_TO_BE_CONTINUED = 2
    PARTIAL_LAST_COMPLETED = 3
    PARTIAL_LAST_FAILED = 4
    CANCELLED = 5
    MEANINGLESS = -1


class Job:
    """A job in the system.

    This follows the fields of the `Standard Workload Format
    <https://www.cs.huji.ac.il/labs/parallel/workload/swf.html>`_ with a couple
    of helper methods to compute slowdown and bounded slowdown of a job. The
    initializer arguments follow the same ordering and have the same meaning
    than those in the SWF description.

    This makes use of the :class:`schedgym.resource.Resource` class to keep
    track of the assigned resources to the job. Resource assignment itself is
    performed by
    :func:`schedgym.scheduler.scheduler.Scheduler.assign_schedule`.

    The figure below shows the relationship between jobs, resources, and the
    basic data structure for resource management (`IntervalTree`).

    .. image:: /img/job-resource.svg
    """

    resources: Resource

    SWF_JOB_MAP = {
        'jobId': 'id',
        'submissionTime': 'submission_time',
        'waitTime': 'wait_time',
        'runTime': 'execution_time',
        'allocProcs': 'processors_allocated',
        'avgCpuUsage': 'average_cpu_use',
        'usedMem': 'memory_use',
        'reqProcs': 'requested_processors',
        'reqTime': 'requested_time',
        'reqMem': 'requested_memory',
        'status': 'status',
        'userId': 'user_id',
        'groupId': 'group_id',
        'executable': 'executable',
        'queueNum': 'queue_number',
        'partNum': 'partition_number',
        'precedingJob': 'preceding_job_id',
        'thinkTime': 'think_time',
    }

    def __init__(
        self,
        job_id=-1,
        submission_time=-1,
        execution_time=-1,
        processors_allocated=-1,
        average_cpu_use=-1,
        memory_use=-1,
        requested_processors=-1,
        requested_time=-1,
        requested_memory=-1,
        status=-1,
        user_id=-1,
        group_id=-1,
        executable=-1,
        queue_number=-1,
        partition_number=-1,
        preceding_job_id=-1,
        think_time=-1,
        wait_time=-1,
        ignore_memory=True,
    ):
        self.id: int = job_id
        self.submission_time: int = submission_time
        self.execution_time: int = execution_time
        self.requested_time: int = requested_time
        self.requested_processors: int = requested_processors
        self.processors_allocated: int = processors_allocated
        self.average_cpu_use: int = average_cpu_use
        self.memory_use: int = memory_use
        self.requested_memory: int = requested_memory
        self.status: JobStatus = status
        self.user_id: int = user_id
        self.group_id: int = group_id
        self.executable: int = executable
        self.queue_number: int = queue_number
        self.partition_number: int = partition_number
        self.preceding_job_id: int = preceding_job_id
        self.think_time = think_time
        self.wait_time = wait_time

        self.resources = Resource()
        self.first_scheduling_promise: int = -1
        self.start_time: int = -1
        self.finish_time: int = -1
        self.ignore_memory = ignore_memory
        self.slot_position: int = -1
        self.free_processors = -1
        self.queued_work = -1
        self.queue_size = -1

    def __str__(self):
        return (
            f'Job<{self.id}, {self.status.name}, start={self.start_time}, '
            f'processors={self.requested_processors}, '
            f'memory={self.requested_memory} '
            f'duration={self.execution_time}>'
        )

    __repr__ = __str__

    @property
    def proper(self):
        """Checks whether this job is a proper job with assigned resources.

        Returns:
            bool: True if the job is proper, and False otherwise.
        """
        processors, memory = self.resources.measure()
        return processors == self.requested_processors and (
            self.ignore_memory or memory == self.requested_memory
        )

    @property
    def slowdown(self):
        """Computes the slowdown of the current job."""
        if self.finish_time < 0:
            warnings.warn(
                f'Failed to obtain slowdown for job {self}. '
                'It may not have finished yet.'
            )
            return -1
        return (
            self.finish_time - self.submission_time
        ) / self.execution_time

    @property
    def bounded_slowdown(self):
        """Gives the bounded slowdown of a job"""
        if self.finish_time < 0:
            warnings.warn(
                f'Failed to obtain avg bounded slowdown for job {self}.'
                'It may not have finished yet.'
            )
            return -1
        return max(
            1,
            (self.finish_time - self.submission_time)
            / max(10, self.execution_time),
        )

    @property
    def swf(self):
        """Returns an SWF representation of this job"""
        return (
            f'{self.id} {self.submission_time} {self.wait_time} '
            f'{self.execution_time} {self.processors_allocated} '
            f'{self.average_cpu_use} '
            f'{self.memory_use} {self.requested_processors} '
            f'{self.requested_time} {self.requested_memory} '
            f'{self.swfstatus} {self.user_id} {self.group_id} '
            f'{self.executable} {self.queue_number} '
            f'{self.partition_number} {self.preceding_job_id} '
            f'{self.think_time}'
        )

    @property
    def swfstatus(self):
        """Returns the job status in the format expected by the SWF."""
        if self.status == JobStatus.COMPLETED:
            return SwfJobStatus.COMPLETED
        return SwfJobStatus.MEANINGLESS

    @staticmethod
    def from_swf_job(swf_job):
        """Converts an SWF job to our internal job format."""
        new_job = Job()
        for key, value in Job.SWF_JOB_MAP.items():
            tmp = getattr(swf_job, key)
            setattr(new_job, value, int(tmp) if 'time' in value else tmp)

        new_job.status = JobStatus.SUBMITTED
        new_job.requested_processors = new_job.processors_allocated
        if new_job.requested_time == -1:
            new_job.requested_time = new_job.execution_time

        return new_job

    @property
    def state(self):
        return JobState(
            self.submission_time,
            self.requested_time,
            self.requested_memory,
            self.requested_processors,
            self.queue_size,
            self.queued_work,
            self.free_processors,
            0,
        )


class JobParameters:
    """Class for using with generative models for job creation.

    Assumes two types of jobs:
        1. "Small" jobs and
        2. "Large" jobs

    A job has probability s of being small and (1-s) of being large.

    Moreover, jobs have a dominant resource to distinguish between CPU-bound
    and I/O bound jobs, with probability of being either CPU-bound and I/O
    bound
    0.5.

    A user of this class must specify all bounds.

    Parameters
    ----------
        lower_time_bound : int
            The minimum time a job will run for
        upper_time_bound : int
            The maximum time a job will run for
        lower_cpu_bound : int
            The minimum number of processors a job will consume
        upper_cpu_bound : int
            The maximum number of processors a job will consume
        lower_mem_bound : int
            The minimum amount of memory a job will consume
        upper_mem_bound : int
            The maximum amount of memory a job will consume

    Used by :class:`schedgym.workload.distribution.BinomialWorkloadGenerator`.
    """

    lower_time_bound: int
    upper_time_bound: int
    lower_resource_bound: int
    upper_resource_bound: int

    @staticmethod
    def _validate_parameters(*args):
        for param in args:
            if param <= 0:
                raise AssertionError(
                    'Unable to work with non-positive bounds.'
                )

    def __init__(
        self,
        lower_time_bound: int,
        upper_time_bound: int,
        lower_cpu_bound: int,
        upper_cpu_bound: int,
        lower_mem_bound: int,
        upper_mem_bound: int,
    ):
        self._validate_parameters(
            lower_time_bound,
            upper_time_bound,
            lower_cpu_bound,
            upper_cpu_bound,
            lower_mem_bound,
            upper_mem_bound,
        )

        self.lower_time_bound = lower_time_bound
        self.upper_time_bound = upper_time_bound
        self.lower_cpu_bound = lower_cpu_bound
        self.upper_cpu_bound = upper_cpu_bound
        self.lower_mem_bound = lower_mem_bound
        self.upper_mem_bound = upper_mem_bound

        self.resource_samplers = {
            PrimaryResource.CPU: lambda: random.randint(
                self.lower_cpu_bound, self.upper_cpu_bound
            ),
            PrimaryResource.MEMORY: lambda: random.randint(
                self.lower_mem_bound, self.upper_mem_bound
            ),
        }

        self.job_id = 1
        self.time_step = 0

    def add_time(self, steps: int = 1) -> None:
        """Increments time in the internal counter."""
        if steps < 0:
            raise AssertionError("Time can't be negative.")
        self.time_step += steps

    def sample(self, submission_time: int = 0) -> Job:
        """Samples a new job.

        Parameters
        ----------
            submission_time : int
                The time at which the new sampled job would have been
                submitted. If omitted, the current times step is used.
        """
        time_duration = random.randint(
            self.lower_time_bound, self.upper_time_bound
        )

        cpu = self.resource_samplers[PrimaryResource.CPU]()
        mem = self.resource_samplers[PrimaryResource.MEMORY]()

        job = Job(
            self.job_id,
            submission_time if submission_time else self.time_step,
            time_duration,
            cpu,
            0,
            mem,
            cpu,
            time_duration,
            mem,
            JobStatus.WAITING,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
        )
        self.job_id += 1

        return job
