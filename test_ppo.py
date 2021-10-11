import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import gym_super_mario_bros
import torch
from Utilities.metrics import MetricLogger
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from env.MadMario.wrappers import RepeatAction, ResizeShape

from pathlib import Path
from agent.PPO import PPO
from env.ParallelEnv.ParallelEnv import parallelEnv
from Utilities.Replay_Buffer.ParallelEnv_buffer import Parallel_buffer
from torch.distributions import Categorical

torch.backends.cudnn.benckmark = True


def test():
    config = {"gamma": 0.95, "step": 0.0001, "img_stack": 4, "action_dim": 2, "useExit": True,
              "clip": 0.2, "num_steps": 800, "num_epochs": 10, "batch_nums": 16,
              "save_path_local": "D:/PythonPro/Myproject/RLproject/ModelPara/Mario_PPO"
              }

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    logger = MetricLogger(save_dir)
    agent = PPO(config=config)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    env = JoypadSpace(env, [['right'], ['right', 'A']])
    env = RepeatAction(env, 4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeShape(env, size=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    episodes = 300
    for e in range(episodes):

        state = env.reset()

        while True:

            env.render()

            values, actions, action_probs = agent.choose_action(np.array([state]))
            next_state, reward, done, info = env.step(actions.item())

            state = next_state
            logger.log_step(reward, loss=0, q=0)

            if done or info['flag_get']:
                break

        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=0,
                step=0
            )

if __name__ == '__main__':
    test()