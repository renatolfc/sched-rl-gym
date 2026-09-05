import numpy as np
import gymnasium.spaces

from ..job import Job
from .base import BaseRmEnv

import logging

logger = logging.getLogger(__name__)


MAXIMUM_JOB_LENGTH = 15
RESOURCE_SLOTS = 10
MAXIMUM_JOB_SIZE = 10

AMOUNT_OF_MEMORY = 10
NUMBER_OF_RESOURCES = 2
NUMBER_OF_PROCESSORS = 10
MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10

NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8

DEFAULT_WORKLOAD = {
    "type": "deeprm",
    "new_job_rate": NEW_JOB_RATE,
    "max_job_size": MAXIMUM_JOB_SIZE,
    "max_job_len": MAXIMUM_JOB_LENGTH,
    "small_job_chance": SMALL_JOB_CHANCE,
}


class CompactRmEnv(BaseRmEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.memory = kwargs.get("memory", AMOUNT_OF_MEMORY)
        self.processors = kwargs.get("processors", NUMBER_OF_PROCESSORS)

        self.renderer = kwargs.get("renderer", None)

        self.maximum_work = self.processors
        self.maximum_work_mem = self.memory

        self._state_buffer_size = (
            self.time_horizon + self.time_horizon * (1 if self.ignore_memory else 2) * 2
            + self.job_slots * 8
            + 1
            + 3
            + 4
        )
        self._state_buffer = np.zeros(self._state_buffer_size, dtype=np.float64)

        self._setup_spaces()

    def _setup_spaces(self):
        self.action_space = gymnasium.spaces.Discrete(self.job_slots + 1)

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_buffer_size,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self.maximum_work = np.log(self.time_limit) * self.processors
        self.maximum_work_mem = np.log(self.time_limit) * self.memory
        return super().reset(seed=seed, options=options)

    def really_done(self) -> bool:
        return (
            len(self.scheduler.queue_admission) == 0 and
            len(self.scheduler.queue_waiting) == 0
        )

    def step(self, action: int):
        done = False
        found = True
        should_be_done = False
        if not (0 <= action < self.action_space.n - 1):
            found = False

        try:
            intermediate = self.simulator.rl_step(
                action if found else None, self.reward_mapper[self.reward_jobs]
            )
        except StopIteration:
            intermediate = [[Job()]]
            should_be_done = True
            done = True

        reward = self.reward if any(intermediate) else 0
        time_exceeded = bool(self.time_limit) and (
            self.scheduler.current_time > self.time_limit
        )
        done = time_exceeded or done or self.really_done()

        if not done and self.smdp and any(intermediate) and not should_be_done:
            rewards = [self.compute_reward(js) for js in intermediate]
            if len(rewards) > 1:
                rewards[0] = 0
            reward = (self.gamma ** np.arange(len(intermediate))).dot(rewards)

        return (self.state, reward, done, False, self.stats if done else {})

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(self.time_horizon, self.job_slots, self.smdp)
        
        snapshots = len(state[0])
        
        offset_idx = 0 if self.smdp else -1
        proc_idx = 1 if self.smdp else 0
        mem_idx = 2 if self.smdp else 1
        
        self._state_buffer.fill(0)
        
        ptr = 0
        if self.smdp:
            self._state_buffer[ptr : ptr + snapshots] = np.log(np.array(state[0]) + 1.0) / np.log(self.time_limit)
            ptr += snapshots

        self._state_buffer[ptr : ptr + snapshots * 2] = (
            np.array([(e[0], e[1]) for e in state[proc_idx]], dtype=np.float32).reshape((-1,))
            / self.processors
        )
        ptr += snapshots * 2
        
        if not self.ignore_memory:
            self._state_buffer[ptr : ptr + snapshots * 2] = (
                np.array([(e[0], e[1]) for e in state[mem_idx]], dtype=np.float32).reshape(
                    (-1,)
                )
                / self.memory
            )
            ptr += snapshots * 2

        jobs_flat = self._normalize_jobs(jobs).reshape((-1,))
        jobs_end = ptr + len(jobs_flat)
        self._state_buffer[ptr:jobs_end] = jobs_flat
        ptr = jobs_end

        backlog_end = ptr + 1
        self._state_buffer[ptr] = backlog / self.backlog_size
        ptr = backlog_end

        running = [
            j
            for j in self.scheduler.queue_running
            if j.start_time + j.requested_time > self.scheduler.current_time
        ]

        remaining_work = (
            sum(
                [
                    np.log(max(j.start_time + j.requested_time - self.scheduler.current_time, 1))
                    * j.requested_processors
                    for j in running
                ]
            )
            / self.maximum_work
        ) if running and self.maximum_work else 0.0
        
        remaining_work_mem = (
            sum(
                [
                    np.log(max(j.start_time + j.requested_time - self.scheduler.current_time, 1))
                    * j.requested_memory
                    for j in running
                ]
            )
            / self.maximum_work_mem
        ) if running and self.maximum_work_mem else 0.0

        queue_size = min(len(self.scheduler.queue_admission) / 1000.0, 1.0)
        time_left = 1 - np.log(self.scheduler.current_time + 1) / np.log(self.time_limit)

        try:
            next_free = min(running, key=lambda x: x.start_time + x.requested_time)
            next_free_arr = np.array(
                (
                    np.log(
                        max(next_free.start_time
                        + next_free.requested_time
                        - self.scheduler.current_time, 1)
                    )
                    / np.log(self.time_limit),
                    next_free.requested_processors / self.processors,
                    (state[proc_idx][0][0] + next_free.requested_processors) / self.processors,
                )
            )
        except ValueError:
            next_free_arr = np.array((0, 0, 1.0))

        next_free_end = ptr + 3
        self._state_buffer[ptr:next_free_end] = next_free_arr
        ptr = next_free_end
        
        self._state_buffer[ptr : ptr + 4] = (
            remaining_work,
            remaining_work_mem,
            queue_size,
            time_left,
        )
        
        # Ensure we return exactly self._state_buffer_size elements, we can pad with 0
        return self._state_buffer[:self._state_buffer_size]

    def _normalize_jobs(self, jobs):
        def _sumdiv(arr, idx, orig, limit):
            arr[idx] = (orig + 1) / (limit + 1)

        ret = np.zeros((len(jobs), len(jobs[0])), dtype=np.float32)
        for i, job in enumerate(jobs):
            _sumdiv(
                ret[i],
                0,
                np.sign(job.submission_time) * np.log(np.abs(job.submission_time) + np.e) - 1,
                np.log(self.time_limit)
            )
            _sumdiv(
                ret[i],
                1,
                np.sign(job.requested_time) * np.log(np.abs(job.requested_time) + np.e) - 1,
                np.log(self.time_limit)
            )
            _sumdiv(ret[i], 2, job.requested_memory, self.memory)
            _sumdiv(ret[i], 3, job.requested_processors, self.processors)
            _sumdiv(ret[i], 4, job.queue_size, self.time_limit)
            _sumdiv(
                ret[i],
                5,
                np.sign(job.queued_work) * np.log(np.abs(job.queued_work) + np.e) - 1,
                np.log(self.time_limit * self.processors),
            )
            _sumdiv(ret[i], 6, job.free_processors, self.processors)
            if len(job) > 7:
                ret[i][7] = job.can_schedule_now
        return ret
