import scipy.optimize as opt
import environment.transportEnv as transport
import matplotlib.pyplot as plt
from utils.prioritised_experience_replay import PrioritizedReplayBuffer
import numpy as np
import pandas as pd
import pickle


class ReplayBufferPER(PrioritizedReplayBuffer):
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        super(ReplayBufferPER, self).__init__(size, 1)

    def store(self, obs, act, rew, next_obs, done):
        super(ReplayBufferPER, self).add(obs, act, rew, next_obs, done, 1)

    def sample_batch(self, batch_size=32):
        obs1, acts, rews, obs2, done, gammas, weights, idxs = super(ReplayBufferPER, self).sample(batch_size, .9)
        return dict(obs1=obs1,
                    obs2=obs2,
                    acts=acts,
                    rews=rews,
                    done=done), [weights, idxs]


class Test():
    def __init__(self):
        self.done_flag = False
        self.env = transport.transportENV()
        self.replay_buffer = ReplayBufferPER(0, 0, size=1e4)
        self.reset()
        self.o2 = None
        self.off_set = np.array([0, 0])
        self.a = np.array([0, 0])
        self.action_list = []
        self.reward_list = []
        self.episode_nummer = 0

    def reset(self):
        while True:
            self.o, self.r, self.d = self.env.reset(), 0, False
            if (self.env.reward) > .3:
                break
        self.done_flag = False
        # print('reset: ', self.o)

    def objective(self, a):
        self.a = np.array(a)*1e-3
        delta_a = self.a - self.off_set


        self.o2, self.r, self.d, _ = self.env.step(delta_a)
        if not(self.d):
            self.replay_buffer.store(self.o, delta_a, self.r, self.o2, self.d)
            self.action_list.append(delta_a)
            self.reward_list.append(self.r)
        elif self.done_flag and self.d:
            pass
        else:
            self.done_flag = True
            self.replay_buffer.store(self.o, delta_a, self.r, self.o2, self.d)
            self.action_list.append(delta_a)
            self.reward_list.append(self.r)

        self.o = self.o2
        self.off_set = self.a
        return -self.r

    def store(self, name):
        file_out = open(name, 'wb')
        pickle.dump(self.replay_buffer._storage, file_out)
        # pd.DataFrame(self.replay_buffer._storage).to_csv('hello.csv')

    def plot(self):
        plt.plot(np.array(self.action_list))
        ax = plt.gca().twinx()
        ax.plot(np.array(self.reward_list), c='lime')
        plt.show()


test = Test()


def constr1(x):
    return 1 - abs(x[1])


def constr2(x):
    return 1 - abs(x[0])

def random_walk(objective, steps=10):
    for _ in range(steps):
        action = np.random.random(2)
        objective(action)
    return None



for _ in range(20):
    test.reset()
    # res = opt.fmin_powell(test.objective, [0.0, 0.0])
    res=opt.fmin_cobyla(test.objective, [0.0, 0.0], [constr1, constr2], rhobeg=.2)
    # random_walk(test.objective)
    print('Reward: ', test.r)

    # print('Output: ', res)
test.plot()

filename = 'Scan_data.obj'
test.store(filename)

filehandler = open(filename, 'rb')
object = pickle.load(filehandler)

print('Length of scan data is: ', len(object))
