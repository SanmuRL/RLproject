from abc import ABC, abstractmethod
from env.ParallelEnv.init_vec_env import VecEnvWrapper
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import multiprocessing
from multiprocessing import Pipe
from env.MadMario.wrappers import RepeatAction, ResizeShape
import gym_super_mario_bros
import gym
import numpy as np
from contextlib import closing

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        # logger.warn('Render not defined for %s' % self)
        pass

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

def work(remote, work_remote, env_name):
    #work_remote.close()
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, [['right'], ['right', 'A']])
    env = RepeatAction(env, 4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeShape(env, size=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, 4)
    while True:
        env.render()
        cmd, inf = remote.recv()
        if cmd == 'env_info':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'step':
            obs, reward, done, info = env.step(inf.item())
            if done:
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplemented



class parallelEnv(VecEnv):
    def __init__(self, env_nums, env_name):
        self.closed = False
        self.wait = False
        # self.envs = [gym_super_mario_bros.make(env_name) for _ in range(env_nums)]
        # self.envs = [JoypadSpace(env, [['right'], ['right', 'A']]) for env in self.envs]
        # self.envs = [RepeatAction(env, 4) for env in self.envs]
        # self.envs = [GrayScaleObservation(env, keep_dim=False) for env in self.envs]
        # self.envs = [ResizeShape(env, size=84) for env in self.envs]
        # self.envs = [TransformObservation(env, f=lambda x: x / 255.) for env in self.envs]
        # self.envs = [FrameStack(env, 4) for env in self.envs]

        self.remote, self.work_remote = zip(*(Pipe() for _ in range(env_nums)))
        self.processes = [multiprocessing.Process(target=work, args=(work_remote, remote, env_name))
                    for (work_remote, remote) in zip(self.work_remote, self.remote)]
        for process in self.processes:
            process.daemon = True
            process.start()
        for workRemote in self.work_remote:
            pass
            #workRemote.close()
        self.remote[0].send(('env_info', None))
        observation_space, action_space = self.remote[0].recv()
        super().__init__(env_nums, observation_space, action_space)

    def reset(self):
        for remote in self.remote:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remote])

    def step_async(self, actions):
        for i in range(len(self.remote)):
            self.remote[i].send(('step', actions[i]))
        self.wait = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remote]
        self.wait = False
        obs, reward, done, info = zip(*(results))
        return np.array(obs), np.array(reward), np.array(done), info

    def close(self):
        if self.closed:
            return
        if self.wait:
            for remote in self.remote:
                remote.recv()
        for remote in self.remote:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True





