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

DEEP_WORKLOAD_NO_MEM = {
    "type": "deeprm",
    "new_job_rate": 1.0,
    "small_job_chance": 0.5,
    "max_job_len": 10,
    "max_job_size": 10,
    "ignore_memory": True,
}

_COMMON_KWARGS = dict(
    simulation_type="event-based",
    job_num_cap=10,
    time_horizon=20,
    backlog_size=20,
)


def _make_deep_env(**extra):
    return deeprm_env.DeepRmEnv(
        use_raw_state=False,
        workload=DEEP_WORKLOAD,
        **_COMMON_KWARGS,
        **extra,
    )


def _make_compact_env(**extra):
    return compact_env.CompactRmEnv(
        time_limit=20,
        workload=DEEP_WORKLOAD,
        **_COMMON_KWARGS,
        **extra,
    )


def _obs_array(obs):
    if isinstance(obs, tuple):
        return np.concatenate([np.asarray(a).ravel() for a in obs])
    return np.asarray(obs)


def _collect_steps(env, n_steps, seed=42):
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy() if isinstance(obs, np.ndarray) else obs]
    for _ in range(n_steps):
        obs, _rew, _term, _trunc, _info = env.step(0)
        observations.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
    return observations


def _assert_step_obs_valid(test_case, obs, expected_shape, label, ndim=None):
    test_case.assertIsInstance(obs, np.ndarray, msg=f"{label}: obs not ndarray")
    test_case.assertEqual(obs.shape, expected_shape, msg=f"{label}: shape mismatch")
    if ndim is not None:
        test_case.assertEqual(obs.ndim, ndim, msg=f"{label}: wrong ndim")
    flat = obs.ravel()
    test_case.assertFalse(np.any(np.isnan(flat)), msg=f"{label}: NaN detected")
    test_case.assertFalse(np.any(np.isinf(flat)), msg=f"{label}: Inf detected")


def _assert_reset_obs_in_range(test_case, env, obs, label):
    flat = np.asarray(obs).ravel()
    low = float(env.observation_space.low.min())
    high = float(env.observation_space.high.max())
    test_case.assertGreaterEqual(
        float(flat.min()), low, msg=f"{label}: reset obs min < {low}"
    )
    test_case.assertLessEqual(
        float(flat.max()), high, msg=f"{label}: reset obs max > {high}"
    )


class TestObservationEquivalence(unittest.TestCase):
    def test_deeprm_default_multistep(self):
        env = _make_deep_env()
        observations = _collect_steps(env, n_steps=50, seed=7)
        expected_shape = env.observation_space.shape
        reset_obs = observations[0]
        _assert_reset_obs_in_range(self, env, reset_obs, "reset")
        for i, obs in enumerate(observations):
            _assert_step_obs_valid(self, obs, expected_shape, f"step {i}", ndim=2)

    def test_deeprm_raw_state_multistep(self):
        env = deeprm_env.DeepRmEnv(
            use_raw_state=True,
            workload=DEEP_WORKLOAD,
            **_COMMON_KWARGS,
        )
        obs, _ = env.reset(seed=13)
        observations = [obs]
        for _ in range(20):
            obs, _rew, _term, _trunc, _info = env.step(0)
            observations.append(obs)

        expected_spaces = list(env.observation_space.spaces)
        reset_obs = observations[0]
        for k, (arr, space) in enumerate(zip(reset_obs, expected_spaces)):
            flat = np.asarray(arr).ravel()
            self.assertGreaterEqual(
                float(flat.min()), 0.0, msg=f"reset comp {k}: min < 0"
            )
            self.assertLessEqual(float(flat.max()), 1.0, msg=f"reset comp {k}: max > 1")

        for i, obs in enumerate(observations):
            self.assertIsInstance(obs, tuple, msg=f"step {i}: raw obs not a tuple")
            self.assertEqual(
                len(obs),
                len(expected_spaces),
                msg=f"step {i}: tuple length mismatch",
            )
            for k, (arr, space) in enumerate(zip(obs, expected_spaces)):
                arr = np.asarray(arr)
                self.assertEqual(
                    arr.shape, space.shape, msg=f"step {i} comp {k}: shape mismatch"
                )
                flat = arr.ravel()
                self.assertFalse(np.any(np.isnan(flat)), msg=f"step {i} comp {k}: NaN")
                self.assertFalse(np.any(np.isinf(flat)), msg=f"step {i} comp {k}: Inf")

    def test_deeprm_ignore_memory_multistep(self):
        env = deeprm_env.DeepRmEnv(
            use_raw_state=False,
            ignore_memory=True,
            workload=DEEP_WORKLOAD_NO_MEM,
            **_COMMON_KWARGS,
        )
        observations = _collect_steps(env, n_steps=20, seed=99)
        expected_shape = env.observation_space.shape
        _assert_reset_obs_in_range(self, env, observations[0], "reset")
        for i, obs in enumerate(observations):
            _assert_step_obs_valid(self, obs, expected_shape, f"step {i}")
        env_default = _make_deep_env()
        self.assertLess(
            expected_shape[1],
            env_default.observation_space.shape[1],
            msg="ignore_memory should produce narrower observation than default",
        )

    def test_compact_env_multistep(self):
        env = _make_compact_env()
        observations = _collect_steps(env, n_steps=50, seed=3)
        expected_shape = env.observation_space.shape
        _assert_reset_obs_in_range(self, env, observations[0], "reset")
        for i, obs in enumerate(observations):
            _assert_step_obs_valid(self, obs, expected_shape, f"step {i}", ndim=1)

    def test_100_steps_no_crash_no_nan(self):
        envs = [
            ("DeepRmEnv", _make_deep_env()),
            ("CompactRmEnv", _make_compact_env()),
        ]
        for name, env in envs:
            with self.subTest(env=name):
                obs, _ = env.reset(seed=0)
                for step in range(100):
                    try:
                        obs, _rew, term, trunc, _info = env.step(0)
                    except Exception as exc:
                        self.fail(f"{name} step {step} raised {exc!r}")
                    if term or trunc:
                        obs, _ = env.reset(seed=step)
                    flat = _obs_array(obs).ravel()
                    self.assertFalse(
                        np.any(np.isnan(flat)),
                        msg=f"{name} step {step}: NaN in obs",
                    )
                    self.assertFalse(
                        np.any(np.isinf(flat)),
                        msg=f"{name} step {step}: Inf in obs",
                    )

    def test_episode_reset_shape_consistency(self):
        env = _make_deep_env()
        obs, _ = env.reset(seed=17)
        expected_shape = obs.shape

        done = False
        for _ in range(500):
            obs, _rew, term, trunc, _info = env.step(0)
            self.assertEqual(
                obs.shape,
                expected_shape,
                msg="Shape changed mid-episode",
            )
            if term or trunc:
                done = True
                break

        obs_after_reset, _ = env.reset(seed=99)
        self.assertEqual(
            obs_after_reset.shape,
            expected_shape,
            msg="Shape changed after reset()",
        )
        for i in range(10):
            obs, _rew, term, trunc, _info = env.step(0)
            self.assertEqual(
                obs.shape,
                expected_shape,
                msg=f"Shape changed at post-reset step {i}",
            )
            flat = obs.ravel()
            self.assertFalse(np.any(np.isnan(flat)), msg=f"NaN at post-reset step {i}")
            self.assertFalse(np.any(np.isinf(flat)), msg=f"Inf at post-reset step {i}")
            if term or trunc:
                break

        self.assertTrue(
            done,
            msg="Episode never ended within 500 steps",
        )


if __name__ == "__main__":
    unittest.main()
