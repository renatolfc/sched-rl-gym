![sched-rl-gym](https://github.com/renatolfc/sched-rl-gym/workflows/sched-rl-gym/badge.svg)

# sched-rl-gym: Gym environment for HPC job scheduling problems

This is an [OpenAI Gym](environment) for job scheduling based on the environment
defined by [DeepRM](https://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf).

Once the module is registered, it can be used as any other gym environment.

Here's an example of a random agent:

```python
import gym
import lugarrl.envs as deeprm

env = gym.make('DeepRM-v0', use_raw_state=True)
ob = env.reset()

for _ in range(1000):
  env.render()
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  if done:
    observation = env.reset()
env.close()
```
