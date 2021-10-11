import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import gym
from env.MadMario.wrappers import RepeatAction
import torch
from Utilities.metrics_DDPG import MetricLogger

from pathlib import Path
from agent.DDPG import DDPG


env = gym.make('BipedalWalker-v3')

env = RepeatAction(env, 4)

torch.backends.cudnn.benckmark = True

config = {"gamma": 0.9, "step": 0.00015, "capacity": 1000000, "batch_size": 256,
          "observation_dim": 24, "action_dim": 4, "useExit": False, "turn_off_exploration": False,
          "epsilon_decay_denominator": 500, "exploration_cycle_length": 30000,
          "epsilon": 1, "epsilon_decay": 1e-5,
          "save_Q_local": "D:/PythonPro/Myproject/RLproject/ModelPara/DDPG/Mario_localQ_DDPG",
          "save_Q_target": "D:/PythonPro/Myproject/RLproject/ModelPara/DDPG/Mario_targetQ_DDQN",
          "save_Pi_local": "D:/PythonPro/Myproject/RLproject/ModelPara/DDPG/Mario_localPi_DDPG",
          "save_Pi_target": "D:/PythonPro/Myproject/RLproject/ModelPara/DDPG/Mario_targetPi_DDQN"
          }

env.reset()

save_dir = Path('checkpoints/DDPG') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

logger = MetricLogger(save_dir)

agent = DDPG(config=config)
episodes = 100000

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()
        with torch.no_grad():
            action, epsilon = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.add_experience(state, action, reward, next_state, done)
        if len(agent.replay_buffer) >= 5e4:
            q, loss_critic, loss_actor = agent.update()
            logger.log_step(reward, loss_critic, loss_actor, q)

        state = next_state

        if done:
            break
    agent.updateNoise()

    logger.log_episode()

    if e % 5000 == 0:
        agent.locally_save_policy()

    if e % 100 == 0:
        logger.record(
            episode=e,
            epsilon=epsilon,
            step=0
        )
