import unittest

import numpy as np

from schedgym.envs import compact_env, deeprm_env

DEEP_WORKLOAD = {
    "type": "deeprm",
    "new_job_rate": 1.0,
    "small_job_chance": 0.5,
    "max_job_len": 10,
    "max_job_size": 10,
    "ignore_memory": False,
}


class TestObservationBuffers(unittest.TestCase):
    def make_deep_env(self):
        return deeprm_env.DeepRmEnv(
            use_raw_state=False,
            simulation_type="event-based",
            job_num_cap=10,
            time_horizon=20,
            backlog_size=20,
            workload=DEEP_WORKLOAD,
        )

    def make_compact_env(self):
        return compact_env.CompactRmEnv(
            simulation_type="event-based",
            time_limit=20,
            job_num_cap=10,
            time_horizon=20,
            backlog_size=20,
            workload=DEEP_WORKLOAD,
        )

    def collect_steps(self, env, count=10):
        obs, _ = env.reset(seed=123)
        observations = [obs]
        for _ in range(count):
            obs, _, _, _, _ = env.step(0)
            observations.append(obs)
        return observations

    def test_deeprm_snapshot_shape(self):
        env = self.make_deep_env()
        observations = self.collect_steps(env, count=10)
        self.assertEqual(len(observations), 11)
        for obs in observations:
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(obs.ndim, 2)

    def test_compactrm_snapshot_shape(self):
        env = self.make_compact_env()
        observations = self.collect_steps(env, count=10)
        self.assertEqual(len(observations), 11)
        for obs in observations:
            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertEqual(obs.ndim, 1)

    def test_build_current_state_shape(self):
        env = self.make_deep_env()
        env.reset(seed=123)
        state, _, _ = env.scheduler.state(env.time_horizon, env.job_slots)
        current = env.build_current_state(state)
        self.assertEqual(len(current), 2)
        self.assertTrue(all(isinstance(arr, np.ndarray) for arr in current))
        self.assertTrue(all(arr.shape == (env.time_horizon, 10) for arr in current))
        self.assertEqual(np.asarray(current).shape, (2, env.time_horizon, 10))

    def test_convert_state_color_mapping(self):
        env = self.make_deep_env()
        env.reset(seed=123)
        state, jobs, backlog = env.scheduler.state(env.time_horizon, env.job_slots)
        converted = env._convert_state(state, jobs, backlog, 0.5)
        self.assertEqual(len(converted), 4)
        for arr in converted:
            self.assertIsInstance(arr, np.ndarray)
            self.assertTrue(np.issubdtype(arr.dtype, np.floating))
            self.assertGreaterEqual(float(arr.min()), 0.0)
            self.assertLessEqual(float(arr.max()), 1.0)

    def test_pack_observation_shape(self):
        env = self.make_deep_env()
        env.reset(seed=123)
        state, jobs, backlog = env.scheduler.state(env.time_horizon, env.job_slots)
        packed = env.pack_observation(env._convert_state(state, jobs, backlog, 0.5))
        self.assertIsInstance(packed, np.ndarray)
        self.assertEqual(packed.shape, env.observation_space.shape)
        self.assertEqual(packed.ndim, 2)

    def test_compact_state_shape_and_range(self):
        env = self.make_compact_env()
        env.reset(seed=123)
        state = env.state
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.shape, env.observation_space.shape)
        self.assertEqual(len(state), env.observation_space.shape[0])
        self.assertGreaterEqual(
            float(state.min()), float(env.observation_space.low.min())
        )
        self.assertLessEqual(
            float(state.max()), float(env.observation_space.high.max())
        )
