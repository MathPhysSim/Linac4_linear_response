import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from NAF_old.src.naf import NAF
from NAF_old.src.network import Network
from NAF_old.src.statistic import Statistic
from NAF_old.utils_old import get_model_dir, preprocess_conf
from simple_environment_linac4 import simpleEnv

flags = tf.app.flags
conf = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False
logger.setLevel('INFO')

random_seed = 888
# set random seed
tf.set_random_seed(random_seed)
np.random.seed(random_seed)

env = simpleEnv()
env.seed(random_seed)


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

    ax = axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.plot(finals, 'r--')
    ax.plot(starts, c='lime')
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')
    plt.savefig(label + '.pdf')
    # fig.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(-np.array(starts), -np.array(finals), c="g", alpha=0.5, marker=r'$\clubsuit$',
                label="Luck")
    plt.ylim(0, 3)
    plt.title(label)
    plt.show()

def plot_convergence(agent, label):
    losses, vs = agent.losses, agent.vs
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('episodes')

    color = 'tab:blue'
    ax.plot(losses, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    ax1.set_ylim(-2, 1)
    color = 'lime'

    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    plt.savefig(label + 'convergence' + '.pdf')
    plt.show()

def main(_):
    model_dir = get_model_dir(conf,
                              ['is_train', 'random_seed', 'monitor', 'display', 'log_level'])

    preprocess_conf(conf)

    with tf.Session() as sess:
        # networks
        shared_args = {
            'sess': sess,
            'input_shape': env.observation_space.shape,
            'action_size': env.action_space.shape[0],
            'hidden_dims': [16, 16],
            'use_batch_norm': False,
            'use_seperate_networks': False,
            'hidden_w': tf.random_uniform_initializer(-0.05, 0.05),
            'action_w': tf.random_uniform_initializer(-0.05, 0.05),
            'hidden_fn': tf.tanh, 'action_fn': tf.tanh,
            'w_reg': None,
        }

        logger.info("Creating prediction network...")
        pred_network = Network(
            scope='pred_network', **shared_args
        )

        logger.info("Creating target network...")
        target_network = Network(
            scope='target_network', **shared_args
        )

        strategy = None
        discount = 0.999
        batch_size = 10
        learning_rate = 1e-3
        max_steps = 500
        update_repeat = 7
        max_episodes = 700
        tau = 1 - 0.999
        is_train = True
        monitor = False
        display = False

        target_network.make_soft_update_from(pred_network, tau)

        # statistics and running the agent
        stat = Statistic(sess, 'default', model_dir, pred_network.variables, update_repeat)
        agent = NAF(sess, env, strategy, pred_network, target_network, stat,
                    discount, batch_size, learning_rate,
                    max_steps, update_repeat, max_episodes)

        agent.run(monitor, display, is_train)

        # plot the results

        label = 'Orig. NAF on: ' + env.__name__

        plot_convergence(agent=agent,label=label)
        plot_results(env, label)


if __name__ == '__main__':
    directory = "checkpoints/awake_test_1/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for f in os.listdir(directory):
            print('Deleting: ', directory + '/' + f)
            os.remove(directory + '/' + f)
        time.sleep(3)
    tf.app.run()
    plt.show()
