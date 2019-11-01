import os
import time

import gym
import numpy as np
import tensorflow as tf

from spinup.algos.naf import core
from spinup.algos.naf.core import get_vars
from spinup.utils.logx import EpochLogger
import matplotlib.pyplot as plt
from NAF_online.src.network import Network


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for NAF agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.size < batch_size:
            idxs = np.arange(self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""

Normalized Advantage Function (NAF)

Some comments on the current NAF implementation:

Missing functionality: 
- [ ] Separate networks
- [ ] Batch norm
- [ ] Regularization

Changed functionality:
- [ ] Target network to Polyak averaging
- [ ] Init. of networks

"""


def naf(env_fn, normalized_advantage_function=core.mlp_normalized_advantage_function, nafnet_kwargs=dict(), seed=123,
        steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.999,
        polyak=0.999, q_lr=1e-3, batch_size=10, start_steps=1e6, update_repeat=7,
        act_noise=1, max_ep_len=1000, logger_kwargs=dict(), save_freq=10, store_session=None):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        normalized_advantage_function: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        normalized_advantage_function (dict): Any kwargs appropriate for the normalized_advantage_function
            function you provided to NAF.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    with tf.Session() as sess:
        tf.set_random_seed(seed)
        np.random.seed(seed)

        env, test_env = env_fn(), env_fn()
        env.seed(seed)
        test_env.seed(seed)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]



        # sess = tf.Session()
        # Share information about action space with naf architecture
        shared_args = {
            'sess': sess,
            'input_shape': env.observation_space.shape,
            'action_size': act_dim,
            'hidden_dims': [100, 100],
            'use_batch_norm': False,
            'use_seperate_networks': False,
            'hidden_w': tf.random_uniform_initializer(-0.05, 0.05),
            'action_w': tf.random_uniform_initializer(-0.05, 0.05),
            'hidden_fn': tf.tanh, 'action_fn': tf.tanh,
            'w_reg': None,
        }

        nafnet_kwargs['action_space'] = env.action_space

        pred_network = Network(
            scope='pred_network', **shared_args
        )

        target_network = Network(
            scope='target_network', **shared_args
        )

        strategy = None
        tau = 1 - 0.999

        target_network.make_soft_update_from(pred_network, tau)

        # Main outputs from computation graph
        # with tf.variable_scope('main'):
        #     print('generate main network')
        # Inputs to computation graph
        # x_ph, a_ph = core.placeholders(obs_dim, act_dim)  # , obs_dim, None, None)
        # Inputs to computation graph

        x_pred, a_pred, mu_pred, V_pred, Q_pred, P_pred, A_pred = \
            pred_network.x, pred_network.u, pred_network.mu, pred_network.V, \
            pred_network.Q, pred_network.P, pred_network.A

        x_targ, a_targ, mu_targ, V_targ, Q_targ, P_targ, A_targ = \
            target_network.x, target_network.u, target_network.mu, target_network.V, \
            target_network.Q, target_network.P, target_network.A
        # \
        normalized_advantage_function(env.observation_space.shape, act_dim=env.action_space.shape, **nafnet_kwargs,
                                      scope='main')
        # Target networks
        # with tf.variable_scope('target'):
        #     print('generate target network')
        #     # Inputs to computation graph
        # x2_ph, a2_ph, _, V_targ, _, _, _,vars_targ =\
        #     normalized_advantage_function(env.observation_space.shape, act_dim=env.action_space.shape, **nafnet_kwargs,
        #                                    scope='target')

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['target', 'main'])
        print('\nNumber of parameters: \t target: %d, \t main: %d\n' % var_counts)

        # r_ph = core.placeholders(None)[0]  # , obs_dim, None, None)
        # Bellman backup for Q function
        # backup = tf.stop_gradient(r_ph + gamma * V_targ)
        # backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * V_targ)
        # q_loss = tf.reduce_mean(tf.square(Q_pred - backup))
        # q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        # train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main'))

        # for v_main, v_targ in zip(get_vars('main'), get_vars('target')):
        #     print(v_targ, v_main)
        # # Initializing targets to match main variables
        # target_init = tf.group([tf.assign(v_targ, v_main)
        #                         for v_main, v_targ in zip(vars_pred, vars_targ)])
        # # Polyak averaging for target variables (previous soft update)
        # target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
        #                           for v_main, v_targ in zip(vars_pred, vars_targ)])

        with tf.name_scope('optimizer'):
            target_y = tf.placeholder(tf.float32, [None], name='target_y')
            loss = tf.reduce_mean(tf.squared_difference(target_y, tf.squeeze(Q_pred)),
                                  name='loss')
            optim = tf.train.AdamOptimizer(learning_rate=q_lr).minimize(loss)

        # print('optimizer\n', get_vars('optimizer'))
        # for v_main in get_vars('main'):
        #     print(v_main)
        # for v_main in get_vars('target'):
        #     print(v_main)
        # for v in tf.get_default_graph().as_graph_def().node:
        #     print(v.name)
        # print('New:')
        # for var in tf.contrib.framework.get_variables(scope='optimizer'):
        #     print(var)
        # Start tensorflow session

        sess.run(tf.global_variables_initializer())
        # sess.run(target_init)
        target_network.hard_copy_from(pred_network)
        # Setup model saving
        logger.setup_tf_saver(sess, inputs={'x': x_pred, 'a': a_pred},
                              outputs={'mu': mu_pred, 'V': V_pred, 'Q': Q_pred, 'P': P_pred, 'A': A_pred})

        # TODO: change scaling back
        def get_action(o, noise_scale):
            a = sess.run(mu_pred, feed_dict={x_pred: o.reshape(1, -1)})[0]
            a += noise_scale * np.random.randn(act_dim)
            act_limit = 100
            return np.clip(a, -act_limit, act_limit)

        # def test_agent(n=10):
        #     test_env.test_flag = True
        #     for j in range(n):
        #         o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        #         while not (d or (ep_len == max_ep_len)):
        #             # Take deterministic actions at test time (noise_scale=0)
        #             o, r, d, _ = test_env.step(get_action(o, 0))
        #             ep_ret += r
        #             ep_len += 1
        #         logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        #     test_env.test_flag = False

        start_time = time.time()

        o, r, d, ep_ret, ep_len, ep_nr = env.reset(), 0, False, 0, 0, 0
        total_steps = steps_per_epoch * epochs
        prestates = []
        actions = []
        rewards = []
        poststates = []
        terminals = []
        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            """
            Until start_steps have elapsed, randomly sample actions
            from a uniform distribution for better exploration. Afterwards, 
            use the learned policy (with some noise, via act_noise). 
            """
            # # TODO: Same noise as old agent
            # if t > start_steps:
            #     a = get_action(o, 0)
            # else:
            #     act_noise_decay = act_noise / (ep_nr + 1)
            #     a = get_action(o, act_noise_decay)

            # Step the env
            noise_scale = 1 / (ep_nr + 1)
            # action = sess.run(mu_pred, feed_dict={x_pred: [o]})[0] + noise_scale * np.random.randn(act_dim)
            u = pred_network.predict([o])[0]
            action =  u + noise_scale*np.random.randn(act_dim)
            prestates.append(o)
            o2, r, d, _ = env.step(action)

            # poststates.append(o2)
            # rewards.append(r)
            # actions.append(action)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, action, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # TODO: Change to update per step
            """
            Perform all NAF updates.
            """
            for iteration in range(update_repeat):
                if len(rewards) >= batch_size:
                    indexes = np.random.choice(len(rewards), size=batch_size)
                else:
                    indexes = np.arange(len(rewards))

                x_t = np.array(prestates)[indexes]
                x_t_plus_1 = np.array(poststates)[indexes]
                r_t = np.array(rewards)[indexes]
                u_t = np.array(actions)[indexes]


                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {'x_ph': np.array(prestates)[indexes],
                             'x2_ph': np.array(poststates)[indexes],
                             'a_ph': np.array(actions)[indexes],
                             'r_ph': np.array(rewards)[indexes],
                             'd_ph': batch['done']
                             }

                v = target_network.predict_v(x_t_plus_1, u_t)
                target_y_values = gamma * np.squeeze(v) + r_t

                outs = sess.run([
                optim, loss,
                pred_network.Q, pred_network.V, pred_network.A,
                ], {
                target_y: target_y_values,
                pred_network.x: x_t,
                pred_network.u: u_t,
                pred_network.is_train: True})

                logger.store(LossQ=outs[1], QVals=outs[2])
                target_network.soft_update_from(pred_network)
                # sess.run(target_update)

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            if d:
                ep_nr += 1
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # End of epoch wrap-up
            if t > 0 and t % steps_per_epoch == 0:
                batch = replay_buffer.sample_batch(int(25))
                feed_dict = {x_pred: batch['obs1'],
                             a_pred: batch['acts']
                             }
                outs = sess.run([V_pred, mu_pred], feed_dict)
                # plt.plot(np.squeeze(outs[0]))
                # plt.plot(np.squeeze(outs[1]))
                # plt.title(str(act_noise_decay))
                # plt.show()

                # plt.plot(np.squeeze(outs[2]))
                # plt.show()
                epoch = t // steps_per_epoch

                # Save model
                if (epoch % save_freq == 0) or (epoch == epochs - 1):
                    logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                # test_agent()

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                # logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                # logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('QVals', with_min_and_max=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time() - start_time)
                logger.dump_tabular()
                # break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='naf')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    naf(lambda: gym.make(args.env), normalized_advantage_function=core.mlp_normalized_advantage_function,
        nafnet_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
