#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple, defaultdict

import argparse

import gym
import numpy as np
import lugarrl.envs as deeprm

from numpy.lib.stride_tricks import as_strided

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

from tensorboardX import SummaryWriter

SLOTS: int = 10
BACKLOG: int = 60
TIME_LIMIT: int = 50
TIME_HORIZON: int = 20
PARALLEL_WORKERS: int = 20
TRAINING_ITERATIONS: int = 6

Experience = namedtuple(
    'Experience',
    field_names='state action reward next_state done log_prob'.split()
)


class PGNet(nn.Module):
    def __init__(self, env):
        super().__init__()

        action_space = env.action_space.n
        observation_space = env.observation_space

        proc, mem, proc_slots, mem_slots, backlog, time = observation_space
        self.input_height = proc.shape[0]
        self.input_width = proc.shape[1] + mem.shape[1] + \
            proc_slots.shape[0] * proc_slots.shape[2] + \
            mem_slots.shape[0] * mem_slots.shape[2] + \
            backlog.shape[1] + 1 if len(time.shape) < 2 else time.shape[-1]
        self.output_size = action_space

        self.hidden = nn.Linear(self.input_height * self.input_width, 20)
        self.out = nn.Linear(20, self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_height * self.input_width)
        x = F.relu(self.hidden(x))
        scores = self.out(x)
        return F.softmax(scores, dim=1)

    def select_action(self, state, device='cpu'):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self(state)
        mass = Categorical(probs)
        action = mass.sample()
        return action.item(), mass.log_prob(action)


class Callback(object):
    def __call__(self, score):
        raise NotImplementedError


class ReduceLROnPlateau(Callback):
    def __init__(self, patience, rate, args):
        self.patience = patience
        self.args = args
        self.rate = rate
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                print(
                    f'Reducing learning rate from {self.args.lr} '
                    f'to {self.args.lr * self.rate}'
                )
                self.args.lr *= self.rate
        else:
            self.best_score = score
            self.counter = 0


def make_discount_array(gamma, timesteps):
    vals = np.zeros(2 * timesteps - 1)
    vals[timesteps - 1:] = gamma ** np.arange(timesteps)
    return as_strided(
        vals[timesteps - 1:],
        shape=(timesteps, timesteps),
        strides=(-vals.strides[0], vals.strides[0]),
        writeable=False
    )


def setup_environment(envname) -> deeprm.DeepRmEnv:
    env: deeprm.DeepRmEnv = gym.make(envname)

    env.job_slots = SLOTS
    env.time_limit = TIME_LIMIT
    env.backlog_size = BACKLOG
    env.time_horizon = TIME_HORIZON

    return env._configure_environment()


def run_episode(env, model, max_episode_length, device='cpu'):
    trajectory = []
    total_reward = 0
    state = env.reset()
    for i in range(max_episode_length):
        action, log_prob = model.select_action(state, device)
        next_state, reward, done, _ = env.step(action)
        exp = Experience(state, action, reward, next_state, done, log_prob)
        trajectory.append(exp)
        total_reward += reward
        if done:
            break
        state = next_state
    return trajectory


def compute_baselines(trajectories):
    returns = np.zeros((len(trajectories), max((len(traj) for traj in trajectories))))
    for i in range(len(trajectories)):
        tmp = np.array([e.reward for e in trajectories[i]])
        returns[i, :len(tmp)] = tmp
    return returns, returns.mean(axis=0)


def train_one_epoch(rank, args, model, device, loss_queue) -> None:
    """Trains the model for one epoch.

    This uses baselining in the REINFORCE algorithm. There are many ways to
    compute baselines. Examples:

        1. The approach taken by DeepRM in the original paper, in which each
           timestep has its own baseline, which is computed as the average
           return for a trajectory.
        2. Computing a global baseline for each trajectory in which it is the
           average return in that trajectory.
        3. A global baseline computed as the average return over all
           trajectories.

    In this function, we follow 1., but nothing prevents us from using 2 or 3.
    """
    # You might need to divide the learning rate by the number of workers
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    env = setup_environment(args.envname)
    optimizer = optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )

    optimizer.zero_grad()
    trajectories = [run_episode(env, model, args.max_episode_length, device)
                    for _ in range(args.trajectories_per_batch)]

    rewards, baselines = compute_baselines(trajectories)
    baselines_mat = np.array([baselines
                              for i in range(args.trajectories_per_batch)])
    baselines_mat = baselines_mat * (rewards != 0)
    discounts = make_discount_array(args.gamma, rewards.shape[1])
    discounted_returns = (discounts @ rewards.T).T
    advantages = discounted_returns - baselines_mat

    policy_loss = []
    for i, t in enumerate(trajectories):
        for j, e in enumerate(t):
            # policy_loss.append(e.log_prob * advantages[i, j])
            policy_loss.append(e.log_prob * advantages[i, j])

    policy_loss = torch.cat(policy_loss).sum()# / len(trajectories)
    # (-policy_loss).backward()
    (-policy_loss).backward()
    optimizer.step()

    loss_queue.put((
        rank, policy_loss.clone().cpu().data.numpy(),
        advantages.mean(), advantages.std(),
        rewards.mean(), rewards.std(),
        discounted_returns.mean(), discounted_returns.std()
    ))


def build_argument_parser():
    parser = argparse.ArgumentParser(description='DeepRM training')
    parser.add_argument('--epochs', type=int, default=TRAINING_ITERATIONS,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--workers', type=int, default=PARALLEL_WORKERS,
                        metavar='N', help='number of workers to train')
    parser.add_argument('--seed', type=int, default=42,
                        metavar='S', help='random seed to use')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='Learning rate for gradient ascent')
    parser.add_argument('--momentum', type=float, default=0.99, metavar='LR',
                        help='momentum for gradient ascent')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables training with CUDA')
    parser.add_argument('--envname', type=str, default='DeepRM-v0',
                        help='OpenAI Gym environment to use')
    parser.add_argument('--max-episode-length', type=int, default=200,
                        metavar='N', help='Maximum number of timesteps in episode')
    parser.add_argument('--trajectories-per-batch', type=int, default=200,
                        metavar='N', help='Number of trajectories in a batch')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None, metavar='PATH',
                        help='Loads a previously-trained model')
    return parser


def main():
    args = build_argument_parser().parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')
    model = PGNet(setup_environment(args.envname)).to(device)
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
    model.share_memory()

    writer = SummaryWriter()
    loss_queue = mp.Queue()

    callbacks = [ReduceLROnPlateau(50, .5, args)]
    for epoch in range(args.epochs):
        print(f'Current epoch: {epoch}')
        losses = []
        if args.debug:
            train_one_epoch(0, args, model, device, loss_queue)
        else:
            workers = []
            args.lr = args.lr
            for rank in range(args.workers):
                p = mp.Process(
                    target=train_one_epoch,
                    args=(rank, args, model, device, loss_queue)
                )
                p.start()
                workers.append(p)
            for w in workers:
                w.join()

            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            losses, extras = [], defaultdict(list)
            features = 'ard'
            while not loss_queue.empty():
                rank, loss, *extra = loss_queue.get()
                print(
                    f'Loss for worker {rank} on epoch {epoch}: {loss}'
                )
                losses.append(loss)
                for i, feature in enumerate(features):
                    extras[f'{feature}μ'].append(extra[i * 2])
                    extras[f'{feature}σ'].append(extra[i * 2 + 1])
                    writer.add_scalar(f'{feature}μ/{rank}', extra[i * 2], epoch)
                    writer.add_scalar(f'{feature}σ/{rank}', extra[i * 2 + 1], epoch)
            print(
                'Loss for epoch {}: {}±{}'.format(epoch, np.mean(losses), np.std(losses))
            )
            writer.add_scalar('loss', np.mean(losses), epoch)
            for i, feature in enumerate(features):
                writer.add_scalar(f'{feature}μ', np.mean(extras[f'{feature}μ']), epoch)
                writer.add_scalar(f'{feature}σ', np.mean(extras[f'{feature}σ']), epoch)
            writer.add_scalar('α', args.lr, epoch)
        for callback in callbacks:
            callback(np.mean(losses))

        writer.flush()
        torch.save(model.state_dict(), f'checkpoint/policy-{epoch}.pth')

    writer.close()
    torch.save(model.state_dict(), 'policy.pth')


if __name__ == '__main__':
    main()
