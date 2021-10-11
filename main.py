import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from env.MadMario.wrappers import RepeatAction, ResizeShape
import torch
from Utilities.metrics import MetricLogger

from pathlib import Path
from agent.DDQN import DDQN


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

env = JoypadSpace(env, [['right'], ['right', 'A']])
env = RepeatAction(env, 4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeShape(env, size=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

torch.backends.cudnn.benckmark = True

config = {"gamma": 0.9, "step": 0.00015, "capacity": 13000, "batch_size": 36,
          "img_stack": 4, "action_dim": 2, "useExit": True, "turn_off_exploration": False,
          "epsilon_decay_denominator": 500, "exploration_cycle_length": 15000,
          "save_path_local": "D:/PythonPro/Myproject/RLproject/ModelPara/Mario_localQ_DDQN",
          "save_path_target": "D:/PythonPro/Myproject/RLproject/ModelPara/Mario_targetQ_DDQN"
          }

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

logger = MetricLogger(save_dir)

agent = DDQN(config=config)
episodes = 30000

for e in range(12000, episodes):

    state = env.reset()

    while True:

        env.render()

        action, epsilon = agent.choose_action(state, e)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.add_experience(state, action, reward, next_state, done)
        if len(agent.replay_buffer) >= 3e3:
            q, loss = agent.update_local_network()
            agent.soft_target_update(0.01)
            logger.log_step(reward, loss=loss, q=q)

        state = next_state

        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 1000 == 0:
        agent.locally_save_policy()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=epsilon,
            step=0
        )
