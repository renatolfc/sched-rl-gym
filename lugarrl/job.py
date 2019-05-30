#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum

import numpy as np


class PrimaryResource(enum.IntEnum):
    CPU = 0
    MEMORY = 1


class JobStatus(enum.IntEnum):
    SUBMITTED = 0
    RUNNING = 1
    WAITING = 2
    COMPLETED = 3
    SCHEDULED = 4


class SwfJobStatus(enum.IntEnum):
    FAILED = 0
    COMPLETED = 1
    PARTIAL_TO_BE_CONTINUED = 2
    PARTIAL_LAST_COMPLETED = 3
    PARTIAL_LAST_FAILED = 4
    CANCELLED = 5
    MEANINGLESS = -1


class Job(object):
    def __init__(self, job_id, submission_time, execution_time, processors_allocated, average_cpu_use, memory_use,
                 requested_processors, requested_time, requested_memory, status, user_id, group_id, executable,
                 queue_number, partition_number, preceding_job_id, think_time, wait_time):
        self.id = job_id
        self.submission_time = submission_time
        self.execution_time = execution_time
        self.requested_time = requested_time
        self.requested_processors = requested_processors
        self.processors_allocated = processors_allocated
        self.average_cpu_use = average_cpu_use
        self.memory_use = memory_use
        self.requested_memory = requested_memory
        self.status = status
        self.user_id = user_id
        self.group_id = group_id
        self.executable = executable
        self.queue_number = queue_number
        self.partition_number = partition_number
        self.preceding_job_id = preceding_job_id
        self.think_time = think_time
        self.wait_time = wait_time

        # intervals of used procs e.g. [(0-10),(15-20)]
        self.processor_list = []
        self.first_scheduling_promise = None
        self.start_time = None
        self.finish_time = None


class JobParameters(object):
    lower_time_bound: int
    upper_time_bound: int
    lower_resource_bound: int
    upper_resource_bound: int

    def __init__(self, lower_time_bound, upper_time_bound, lower_cpu_bound, upper_cpu_bound, lower_mem_bound,
                 upper_mem_bound):
        self.lower_time_bound = lower_time_bound
        self.upper_time_bound = upper_time_bound
        self.lower_cpu_bound = lower_cpu_bound
        self.upper_cpu_bound = upper_cpu_bound
        self.lower_mem_bound = lower_mem_bound
        self.upper_mem_bound = upper_mem_bound

        self.resource_samplers = {
            PrimaryResource.CPU: lambda: np.random.randint(self.lower_cpu_bound, self.upper_cpu_bound + 1),
            PrimaryResource.MEMORY: lambda: np.random.randint(self.lower_mem_bound, self.upper_mem_bound + 1),
        }

        self.job_id = 1
        self.time_step = 1

    def add_time(self, steps=1):
        self.time_step += steps

    def sample(self, submission_time=0):
        time_duration = np.random.randint(self.lower_time_bound, self.upper_time_bound + 1)

        cpu = self.resource_samplers[PrimaryResource.CPU]()
        mem = self.resource_samplers[PrimaryResource.MEMORY]()

        job = Job(
            self.job_id, submission_time if submission_time else self.time_step, time_duration, cpu, 0, mem, cpu,
            time_duration, mem, SwfJobStatus.COMPLETED, 1, 1, 1, 1, 1, -1, -1, -1
        )
        self.job_id += 1

        return job
