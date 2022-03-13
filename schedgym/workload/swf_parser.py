#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
swf_parser - Parser for the Standard Workload Format (SWF)

A full description of the format, with meanings for each field is available on
the web at http://www.cs.huji.ac.il/labs/parallel/workload/swf.html.
"""

import logging
from enum import IntEnum
from typing import Dict, Generator, Optional, Type, Union, cast

from ..job import Job, SwfJobStatus

logger = logging.getLogger(__name__)  # pylint: disable=C


class SwfFields(IntEnum):
    """Fields of the Standard Workload Format."""

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


CONVERTERS: Dict[SwfFields, Union[Type[int], Type[float]]] = {
    key: int if key != SwfFields.AVG_CPU_USAGE else float for key in SwfFields
}


def parse_int(line: str) -> int:
    return int(line.split(':')[-1].strip())


def parse(
        filename,
        processors: Optional[int] = None,
        memory: int = 0,
        ignore_memory: bool = True
) -> Generator[Job, None, None]:
    """Parser for SWF job files.

    The SWF is a simple format with commented lines starting with the ';'
    character and other lines separated by spaces.

    Parsing, therefore, involves splitting the lines and associating each
    column of the file with a field.
    """

    max_procs: int = 0
    max_nodes: int = 0
    with open(filename, 'r') as fp:  # pylint: disable=C
        for line in fp:
            if line.startswith(';'):
                if line.startswith('; MaxNodes:'):
                    max_nodes = parse_int(line)
                elif line.startswith('; MaxProcs:'):
                    max_procs = parse_int(line)
                continue

            if processors is None:
                if max_procs == 0:
                    if max_nodes == 0:
                        raise ValueError(
                            f"Unable to load trace {filename} "
                            "without a number of processors"
                        )
                    else:
                        processors = max_nodes
                else:
                    processors = max_procs

            fields = [  # Converts all fields according to our rules
                CONVERTERS[SwfFields(i)](f)
                for i, f in enumerate(line.strip().split())
            ]

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
                SwfJobStatus(cast(int, fields[SwfFields.STATUS])),
                fields[SwfFields.USER_ID],
                fields[SwfFields.GROUP_ID],
                fields[SwfFields.EXECUTABLE],
                fields[SwfFields.QUEUE_NUM],
                fields[SwfFields.PART_NUM],
                fields[SwfFields.PRECEDING_JOB],
                fields[SwfFields.THINK_TIME],
                fields[SwfFields.WAIT_TIME],
            )

            if job.requested_memory < 0 < job.memory_use:
                job.requested_memory = job.memory_use

            if job.requested_processors < 0 < job.processors_allocated:
                job.requested_processors = job.processors_allocated

            if job.requested_memory < 0 and ignore_memory:
                job.requested_memory = 0

            if job.requested_processors < 1:
                job.requested_processors = 1

            if job.requested_memory < 1:
                job.requested_memory = 1

            if job.execution_time < 1:
                job.execution_time = 1

            if job.submission_time < 0:
                job.submission_time = 0

            if job.requested_time < job.execution_time:
                job.requested_time = job.execution_time

            if job.requested_processors > processors:
                job.requested_processors = processors

            if job.requested_memory > memory:
                job.requested_memory = memory

            yield job
