#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import lugarrl.envs as deeprm

EPISODES = 1
MAX_EPISODE_LENGTH = 200


def sjf_action(observation):
    "Selects the job SJF (Shortest Job First) would select."
    best = observation[2].shape[1] + 1  # infinity
    best_idx = observation[2].shape[0]
    free_proc = observation[0].shape[1] - np.sum(observation[0][0, :] != 0)
    free_mem = observation[1].shape[1] - np.sum(observation[1][0, :] != 0)
    for slot in range(observation[2].shape[0]):
        used_proc = (observation[2][slot, 0, :] != 0).sum()
        used_mem = (observation[3][slot, 0, :] != 0).sum()
        if used_proc and used_proc <= free_proc and used_mem <= free_mem:
            tmp = np.sum(observation[2][slot, :, 0])
            if tmp < best:
                best_idx = slot
                best = tmp
    return best_idx


def find_position(q, idx):
    for i, j in enumerate(q):
        if j.slot_position == idx:
            return i
    return idx


def pack_observation(ob, time_horizon):
    ob = list(ob)
    ob[2] = np.hstack(ob[2]).reshape((time_horizon, -1))
    ob[3] = np.hstack(ob[3]).reshape((time_horizon, -1))
    return np.hstack(ob)


def main():
    env: deeprm.DeepRmEnv = gym.make('DeepRM-v0')
    env.use_raw_state = True
    time_horizon = env.observation_space[0].shape[0]
    for episode in range(EPISODES):
        ob = env.reset()
        action = sjf_action(ob)
        for _ in range(50):
            ob, reward, done, _ = env.step(action)
            action = sjf_action(ob)
            ob = pack_observation(ob, time_horizon)
            env.render()
            if done:
                break
    import pdb; pdb.set_trace()
    env.close()

if __name__ == '__main__':
    main()
