import logging

import gymnasium.spaces
import numpy as np

from ..job import Job
from .base import BaseRmEnv
from .simulator import DeepRmSimulator
from .workload import DeepRmWorkloadGenerator

logger = logging.getLogger(__name__)

MAXIMUM_QUEUE_SIZE = 16

RESOURCE_SLOTS = 10

NUMBER_OF_RESOURCES = 2
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10


class DeepRmEnv(BaseRmEnv):
    n_work: int
    n_resources: int
    use_raw_sate: bool
    simulator: DeepRmSimulator
    workload: DeepRmWorkloadGenerator
    observation_space: gymnasium.spaces.Tuple | gymnasium.spaces.Box
    action_space: gymnasium.spaces.Discrete

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_raw_state = kwargs.get("use_raw_state", False)

        self.n_resources = kwargs.get(
            "n_resources", NUMBER_OF_RESOURCES
        )  # resources in the system
        self.n_work = kwargs.get(
            "n_work", MAXIMUM_QUEUE_SIZE
        )  # max amount of work in the queue
        if self.backlog_size % self.time_horizon:
            raise AssertionError("Backlog must be a multiple of time horizon")

        self.backlog_width = self.backlog_size // self.time_horizon

        self.setup_spaces()

    def setup_spaces(self):
        self.action_space = gymnasium.spaces.Discrete(self.job_slots + 1)
        if self.use_raw_state:
            self.setup_raw_spaces()
        else:
            self.setup_image_spaces()

    def setup_image_spaces(self):
        total_width = (
            (0 if self.ignore_memory else (self.job_slots + 1))
            * self.scheduler.total_memory
            + (self.job_slots + 1) * self.scheduler.number_of_processors
            + self.backlog_width
            + 1
        )
        self.observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.time_horizon, total_width),
        )
        self._obs_buffer = np.empty((self.time_horizon, total_width), dtype=np.float64)

    def setup_raw_spaces(self):
        self.processor_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.time_horizon, self.scheduler.number_of_processors),
        )
        self.processor_slots_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                self.job_slots,
                self.time_horizon,
                self.scheduler.number_of_processors,
            ),
        )
        self.backlog_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(self.time_horizon, self.backlog_width)
        )
        self.time_since_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(self.time_horizon, 1)
        )

        if self.ignore_memory:
            self.observation_space = gymnasium.spaces.Tuple(
                (
                    self.processor_space,
                    self.processor_slots_space,
                    self.backlog_space,
                    self.time_since_space,
                )
            )
        else:
            self.memory_space = gymnasium.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.time_horizon, self.scheduler.total_memory),
            )
            self.memory_slots_space = gymnasium.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(
                    self.job_slots,
                    self.time_horizon,
                    self.scheduler.total_memory,
                ),
            )
            self.observation_space = gymnasium.spaces.Tuple(
                (
                    self.processor_space,
                    self.memory_space,
                    self.processor_slots_space,
                    self.memory_slots_space,
                    self.backlog_space,
                    self.time_since_space,
                )
            )

        self.observation_space.n = np.sum(  # type: ignore
            [
                np.prod(e.shape) if isinstance(e, gymnasium.spaces.Box) else e.n
                for e in self.observation_space
            ]
        )

    @property
    def state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots, self.smdp
        )
        s = self._convert_state(
            state,
            jobs,
            backlog,
            (
                (self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB
            ),
        )
        if self.use_raw_state:
            current, wait, backlog_arr, time_arr = s
            if self.ignore_memory:
                return (current[0], wait[0], backlog_arr, time_arr)
            return (current[0], current[1], wait[0], wait[1], backlog_arr, time_arr)
        return self.pack_observation(s)

    def pack_observation(self, ob):
        current, wait, backlog, time = ob
        wait = wait.reshape(self.time_horizon, -1)
        current = current.reshape(self.time_horizon, -1)
        col = 0
        w = current.shape[1]
        self._obs_buffer[:, col : col + w] = current
        col += w
        w = wait.shape[1]
        self._obs_buffer[:, col : col + w] = wait
        col += w
        w = backlog.shape[1]
        self._obs_buffer[:, col : col + w] = backlog
        col += w
        self._obs_buffer[:, col : col + 1] = time
        return self._obs_buffer

    def find_slot_position(self, action):
        if action < len(self.scheduler.queue_admission):
            return action
        return self.action_space.n - 1

    def step(self, action: int):
        done = False
        found = False
        should_be_done = False
        if 0 <= action < self.action_space.n - 1:
            action = self.find_slot_position(action)
            found = True
        try:
            intermediate = self.simulator.rl_step(
                action if found else None, self.reward_mapper[self.reward_jobs]
            )
        except StopIteration:
            intermediate = [[Job()]]
            should_be_done = True
            done = True

        reward = self.reward if any(intermediate) else 0
        done = bool(self.time_limit) and (
            self.scheduler.current_time > self.time_limit or done
        )

        if not done and self.smdp and any(intermediate) and not should_be_done:
            rewards = [self.compute_reward(js) for js in intermediate]
            if len(rewards) > 1:
                rewards[0] = 0
            reward = (self.gamma ** np.arange(len(intermediate))).dot(rewards)

        return (self.state, reward, done, False, self.stats if done else {})
