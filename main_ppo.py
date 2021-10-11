import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import gym_super_mario_bros
import torch
from Utilities.metrics_PPO import MetricLogger
import numpy as np

from pathlib import Path
from agent.PPO import PPO
from env.ParallelEnv.ParallelEnv import parallelEnv
from Utilities.Replay_Buffer.ParallelEnv_buffer import Parallel_buffer
from torch.distributions import Categorical

torch.backends.cudnn.benckmark = True


def train():
    config = {"gamma": 0.95, "step": 0.0001, "img_stack": 4, "action_dim": 2, "useExit": True,
              "clip": 0.2, "num_steps": 800, "num_epochs": 10, "batch_nums": 16,
              "save_path_local": "D:/PythonPro/Myproject/RLproject/ModelPara/Mario_PPO"
              }

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    logger = MetricLogger(save_dir)
    rollouts = Parallel_buffer(800, (4, 84, 84), 5)
    agent = PPO(config=config)
    envs = parallelEnv(env_nums=5, env_name='SuperMarioBros-1-1-v0')
    episodes = 2000
    obs = envs.reset()
    for e in range(episodes):

        for i in range(agent.num_steps):
            with torch.no_grad():
                values, actions, action_probs = agent.choose_action(obs)
            obs, reward, done, info = envs.step(actions.cpu().detach().numpy())
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            obs_t = torch.tensor(obs)
            reward_t = torch.tensor(reward).unsqueeze(1)
            rollouts.add_experience(obs_t, actions, action_probs, reward_t, masks, values)

            logger.log_step(torch.mean(values).item())

        with torch.no_grad():
            next_value = agent.policy(rollouts.obs[-1])[0].detach()

        avg_rewards = rollouts.calculate_returns(next_value, agent.gamma)

        sum_loss, l1, l2 = agent.update_policy(rollouts)
        sum_loss /= 160
        l1 /= 160
        l2 /= 160

        rollouts.next_episode()

        logger.log_episode(sum_loss, avg_rewards, l1, l2)
        if e % 60 == 0:
            agent.locally_save_policy()

        logger.record(
            episode=e,
            epsilon=0,
        )

if __name__ == '__main__':
    train()
