import time

import gym
import numpy as np
import tensorflow as tf

from spinup.algos.naf import core
from spinup.algos.naf.core import get_vars
from spinup.utils.logx import EpochLogger


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


def naf(env_fn, normalized_advantage_function=core.mlp_normalized_advantage_function, nafnet_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.999,
        polyak=0.9995, q_lr=1e-4, batch_size=100, start_steps=500, update_repeat=5,
        act_noise=0.001, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
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

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    nafnet_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, target_y = core.placeholders(obs_dim, act_dim, obs_dim, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        print('generate main network')
        mu_pred, V_pred, Q_pred, P_pred, A_pred = normalized_advantage_function(x_ph, a_ph, **nafnet_kwargs)
    # Target networks
    with tf.variable_scope('target'):
        print('generate target network')
        mu_targ, V_targ, Q_targ, P_targ, A_targ = normalized_advantage_function(x2_ph, a_ph, **nafnet_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['target', 'main'])
    print('\nNumber of parameters: \t target: %d, \t total: %d\n' % var_counts)

    with tf.name_scope('optimizer'):
        # NAF losses
        q_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(Q_pred), target_y))
        # Train ops for q
        # q_optim = tf.train.AdamOptimizer(learning_rate=q_lr).minimize(q_loss, var_list=get_vars('main'))
        q_optim = tf.train.AdamOptimizer(learning_rate=q_lr).minimize(q_loss)

    # Polyak averaging for target variables (previous soft update)
    target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    for v_main, v_targ in (zip(get_vars('main'), get_vars('target'))):
        print(v_main, v_targ)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # tf.initialize_all_variables().run(session=sess)
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                          outputs={'mu': mu_pred, 'V': V_pred, 'Q': Q_pred, 'P': P_pred, 'A': A_pred})

    # TODO: change scaling back
    def get_action(o, noise_scale):
        a = sess.run(mu_pred, feed_dict={x_ph: o.reshape(1, -1)})[0]
        value = a + noise_scale * np.random.randn(act_dim)
        value = 1e-3*value
        return np.clip(value, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len, ep_nr = env.reset(), 0, False, 0, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        #TODO: Same noise as old agent
        # if t > start_steps:
        act_noise = act_noise * 1 / (ep_nr * 0.25 + 1)
        a = get_action(o, act_noise)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # TODO: Change to update per step
        if True:
            """
            Perform all NAF updates.
            """
            q_list = []
            v_list = []
            a_list = []
            l_list = []
            for iteration in range(update_repeat):
                batch = replay_buffer.sample_batch(batch_size)
                # Q-learning update

                # 1. Calculate value function
                v = sess.run(V_targ, feed_dict={x2_ph: batch['obs2'],
                                                    a_ph: batch['acts']})

                # TODO: Formulate in Tensorflow?
                # 2. Calculate target value according to Bellman
                target_values = gamma * np.squeeze(v) + np.array(batch['rews'])

                # 3. Feed into Tensorflow
                outs = sess.run([q_optim,  Q_pred, V_pred, A_pred, q_loss,],
                                {target_y: target_values,
                                 x_ph: batch['obs1'],
                                 a_ph: batch['acts']
                                 })
                logger.store(LossQ=outs[4], QVals=outs[2])

                q_list.extend(outs[1])
                v_list.extend(outs[2])
                a_list.extend(outs[3])
                l_list.append(outs[4])

                sess.run(target_update)



        if d or (ep_len == max_ep_len):
            print("t: %s, q: %s, v: %s, a: %s, l: %s" \
                      % (t, np.mean(q_list), np.mean(v_list), np.mean(a_list), np.mean(l_list)))
            # if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            ep_nr += 1

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


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

    naf(lambda: gym.make(args.env), normalized_advantage_function=core.mlp_actor_critic,
        nafnet_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
