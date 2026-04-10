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
        scheduled_count = 0
        for job in self.queue_admission:
            resources = self.can_schedule_now(job)
            if resources:
                self.assign_schedule(job, resources, self.current_time)
                scheduled_count += 1
            else:
                break
        self.queue_admission = self.queue_admission[scheduled_count:]
        self._queued_work_total = sum(
            j.requested_time * j.requested_processors for j in self.queue_admission
        )
