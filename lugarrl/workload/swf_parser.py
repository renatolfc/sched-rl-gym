#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
swf_parser - Parser for the Standard Workload Format (SWF)

A full description of the format, with meanings for each field is available on the web at
http://www.cs.huji.ac.il/labs/parallel/workload/swf.html.
"""

from enum import IntEnum

import logging

from ..job import Job, SwfJobStatus

logger = logging.getLogger(__name__)


class SwfFields(IntEnum):
    JOB_ID = 0
    SUBMITTED = 1
    WAIT_TIME = 2
    EXEC_TIME = 3
    ALLOC_PROCS = 4
    AVG_CPU_USAGE = 5
    USED_MEM = 6
    REQ_PROCS = 7
    REQ_TIME = 8
    REQ_MEM = 9
    STATUS = 10
    USER_ID = 11
    GROUP_ID = 12
    EXECUTABLE = 13
    QUEUE_NUM = 14
    PART_NUM = 15
    PRECEDING_JOB = 16
    THINK_TIME = 17


CONVERTERS = {key: int if key != SwfFields.AVG_CPU_USAGE else float
              for key in SwfFields}


def parse(filename, processors, memory, memory_placeholder=1):
    with open(filename, 'r') as fp:
        for line in fp:
            if ';' in line:
                continue
            fields = line.strip().split()
            fields = [CONVERTERS[SwfFields(i)](f) for i, f in enumerate(fields)]

            job = Job(
                fields[SwfFields.JOB_ID],
                fields[SwfFields.SUBMITTED],
                fields[SwfFields.EXEC_TIME],
                fields[SwfFields.ALLOC_PROCS],
                fields[SwfFields.AVG_CPU_USAGE],
                fields[SwfFields.USED_MEM],
                fields[SwfFields.REQ_PROCS],
                fields[SwfFields.REQ_TIME],
                fields[SwfFields.REQ_MEM],
                SwfJobStatus(fields[SwfFields.STATUS]),
                fields[SwfFields.USER_ID],
                fields[SwfFields.GROUP_ID],
                fields[SwfFields.EXECUTABLE],
                fields[SwfFields.QUEUE_NUM],
                fields[SwfFields.PART_NUM],
                fields[SwfFields.PRECEDING_JOB],
                fields[SwfFields.THINK_TIME],
                fields[SwfFields.WAIT_TIME]
            )

            if job.requested_memory < 0 < job.memory_use:
                job.requested_memory = job.memory_use

            if job.requested_processors < 0 < job.processors_allocated:
                job.requested_processors = job.processors_allocated

            if job.requested_memory < 0 and memory_placeholder:
                job.requested_memory = memory_placeholder

            if (job.requested_processors < 1 or job.requested_memory < 1
                    or job.execution_time < 1 or job.submission_time < 0):
                logger.warning(f'Ignoring malformed job {job.id}')
                continue

            if job.requested_time < job.execution_time:
                job.requested_time = job.execution_time

            if job.requested_processors > processors:
                job.requested_processors = processors

            if job.requested_memory > memory:
                job.requested_memory = memory

            yield job


if __name__ == '__main__':
    import sys
    for j in parse(sys.argv[1], 1024, 1024):
        print(j)
