# For the moment only basic functionality is provided (select a full plane)


# matplotlib.use("Qt5Agg")
import os
import time

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# import PyQt5
from simple_environment_linac4 import simpleEnv
from spinup import td3, ddpg, sac, ppo


def plot_results(env, label):
    # plotting
    print('now plotting')
    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    finals = []
    starts = []

    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            finals.append(rewards[i][len(rewards[i]) - 1])
            starts.append(-np.sqrt(np.mean(np.power(initial_states[i], 2))))
            iterations.append(len(rewards[i]))

    plot_suffix = f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax=axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.plot(finals, 'r--')
    ax.plot(starts, c='lime')
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')
    plt.savefig(label+'.pdf')
    # fig.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(-np.array(starts), -np.array(finals), c="g", alpha=0.5, marker=r'$\clubsuit$',
                label="Luck")
    plt.ylim(0, 3)
    plt.title(label)
    plt.show()

env = simpleEnv()
random_seed = 888
# set random seed

tf.set_random_seed(random_seed)
np.random.seed(random_seed)

env.seed(random_seed)
env.reset()
env_fn = lambda: env

ac_kwargs = dict()#dict(hidden_sizes= (16, 16))

# directory_naf = "logging/awake/NAF"
# if not os.path.exists(directory_naf):
#     os.makedirs(directory_naf)
# else:
#     for f in os.listdir(directory_naf):
#         print('Deleting: ', directory_naf + '/' + f)
#         os.remove(directory_naf + '/' + f)
#     time.sleep(3)
output_dir = 'logging/awake/NAF/'

logger_kwargs = dict(output_dir=output_dir, exp_name='transport_awake')

agent = ddpg(env_fn=env_fn, epochs=10, steps_per_epoch=100, ac_kwargs=ac_kwargs,
            logger_kwargs=logger_kwargs, start_steps=1e6, seed=random_seed)

plot_name = 'Stats'
name = plot_name
data = pd.read_csv(output_dir + '/progress.txt', sep="\t")

data.index = data['TotalEnvInteracts']
data_plot = data[['EpLen', 'MinEpRet', 'AverageEpRet']]
data_plot.plot(secondary_y=['MinEpRet', 'AverageEpRet'])

label = 'Classic DDPG on: ' + env.__name__
plt.title(label=label)
plt.ylim(-10,0)
# plt.savefig(name + '.pdf')
plt.show()


plot_results(env, label)