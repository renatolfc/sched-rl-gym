from collections.abc import Sequence

from schedgym.job import Job


class TraceCursor:
    def __init__(self, jobs: Sequence[Job]) -> None:
        self.jobs = jobs
        self.index = 0

    def validate_monotonic(self) -> None:
        for previous, current in zip(self.jobs, self.jobs[1:]):
            if previous.submission_time > current.submission_time:
                raise ValueError(
                    "Trace jobs must be sorted by nondecreasing submission_time"
                )

    def has_next(self) -> bool:
        return self.index < len(self.jobs)

    @staticmethod
    def effective_submission_time(job: Job) -> int:
        return max(1, job.submission_time)

    def next_time(self) -> int | None:
        if not self.has_next():
            return None
        return self.effective_submission_time(self.jobs[self.index])

    def pop_jobs_at(self, time: int) -> list[Job]:
        start = self.index
        while (
            self.index < len(self.jobs)
            and self.effective_submission_time(self.jobs[self.index]) == time
        ):
            self.index += 1
        return list(self.jobs[start : self.index])

    def remaining(self) -> int:
        return len(self.jobs) - self.index
