#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import gym
import json
import numpy as np
import schedgym.envs as deeprm

EPISODES = 1
MAX_EPISODE_LENGTH = 200


def sjf_action(observation):
    "Selects the job SJF (Shortest Job First) would select."

    current, wait, _, _ = observation
    best = wait.shape[2] + 1  # infinity
    best_idx = wait.shape[1]

    free = np.ones(current.shape[0]) * current.shape[-1] - np.sum(current[:, 0, :] != 0)

    for slot in range(wait.shape[1]):
        required_resources = (wait[:, slot, 0, :] != 0).sum(axis=1)
        if np.all(required_resources) and np.all(required_resources <= free):
            tmp = np.sum(wait[0, slot, :, 0])
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
    current, wait, backlog, time = ob
    wait = wait.reshape(time_horizon, -1)
    current = current.reshape(time_horizon, -1)
    return np.hstack((current, wait, backlog, time))

def main():
    kwargs = {'use_raw_state': True}
    if os.path.exists('config/test.json'):
        with open('config/test.json', 'r') as fp:
            kwargs = json.load(fp)
    env: deeprm.DeepRmEnv = gym.make('DeepRM-v0', **kwargs)
    time_horizon = env.reset()[0].shape[1]
    for episode in range(EPISODES):
        ob = env.reset()
        action = sjf_action(ob)
        while True:
            ob, reward, done, _ = env.step(action)
            action = sjf_action(ob)
            ob = pack_observation(ob, time_horizon)
            env.render()
            if done:
                break
    env.close()

if __name__ == '__main__':
    main()
