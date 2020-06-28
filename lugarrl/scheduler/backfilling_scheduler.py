#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional

from lugarrl.job import Job, JobStatus
from lugarrl.scheduler import Scheduler
from lugarrl.event import JobEvent


class BackfillingScheduler(Scheduler):
    "EASY backfilling scheduler"

    reservation: Optional[Tuple[JobEvent, JobEvent]]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reservation = None

    def handle_reservation(self) -> None:
        if not self.reservation:
            return

        start, finish = self.reservation
        if (start.time == self.current_time
                or start.job.status != JobStatus.WAITING):
            # Reservation will be fulfilled
            self.reservation = None
            return

        resources = self.can_schedule_now(start.job)
        if resources:
            self.queue_waiting.remove(start.job)

            self.job_events.remove(start)
            self.job_events.remove(finish)

            self.assign_schedule(
                start.job, resources, self.current_time
            )
            self.reservation = None

    def schedule(self) -> None:
        ignored_jobs: List[Job] = []

        self.handle_reservation()
        for job in self.queue_admission:
            resources = self.can_schedule_now(job)
            if resources:
                self.assign_schedule(job, resources, self.current_time)
            else:
                if not self.reservation:
                    # This is the first job without a reservation.
                    # We're doing EASY backfilling, so we create a
                    # reservation for this one job and keep going
                    time, resources = self.find_first_time_for(job)
                    if not resources:
                        raise AssertionError("Something is terribly wrong")
                    self.reservation = self.assign_schedule(
                        job, resources, time
                    )
                else:
                    # We already have a reservation, so we skip this job
                    ignored_jobs.append(job)
        self.queue_admission = ignored_jobs
