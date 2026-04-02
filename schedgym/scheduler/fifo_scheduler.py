"""fifo_scheduler - First-In First-Out module"""

from schedgym.job import Job
from schedgym.scheduler import Scheduler


class FifoScheduler(Scheduler):
    """A FIFO scheduler."""

    def schedule(self) -> None:
        """Schedules jobs according to submission time.

        This implements a *string* FIFO strategy, meaning it will always obey
        submission order, even when it creates fragmentation.
        """
        scheduled_jobs: list[Job] = []
        for job in self.queue_admission:
            resources = self.can_schedule_now(job)
            if resources:
                self.assign_schedule(job, resources, self.current_time)
                scheduled_jobs.append(job)
            else:
                break
        for job in scheduled_jobs:
            self.queue_admission.remove(job)
