from logging import getLogger

from NAF_old.src.utils import get_timestamp

logger = getLogger(__name__)

import numpy as np
import tensorflow as tf


class NAF(object):
    def __init__(self, sess,
                 env, strategy, pred_network, target_network, stat,
                 discount, batch_size, learning_rate,
                 max_steps, update_repeat, max_episodes, private_settings=False):
        self.losses = []
        self.vs = []
        self.sess = sess
        self.env = env
        self.strategy = strategy
        self.pred_network = pred_network
        self.target_network = target_network
        self.stat = stat

        self.discount = discount
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.action_size = env.action_space.shape[0]

        self.max_steps = max_steps
        self.update_repeat = update_repeat
        self.max_episodes = max_episodes

        self.prestates = []
        self.actions = []
        self.rewards = []
        self.poststates = []
        self.terminals = []

        with tf.name_scope('optimizer'):
            self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(self.pred_network.Q)),
                                       name='loss')
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def run(self, monitor=False, display=False, is_train=True):
        print('Training:', is_train)
        self.stat.load_model()  # including init
        self.target_network.hard_copy_from(self.pred_network)

        if monitor:
            self.env.monitor.start('/tmp/%s-%s' % (self.stat.env_name, get_timestamp()))
        for self.idx_episode in range(self.max_episodes):
            state = self.env.reset()
            self.learning_rate = self.learning_rate  # /((self.idx_episode+1.)*5)

            for t in range(0, self.max_steps):
                # 1. predict
                action = self.predict(state, is_train)
                # 2. step
                self.prestates.append(state)
                state, reward, terminal, _ = self.env.step(action)
                self.poststates.append(state)
                terminal = True if t == self.max_steps - 1 else terminal
                # 3. perceive
                if is_train:
                    self.rewards.append(reward)
                    self.actions.append(action)
                    q, v, a, l = self.perceive()
                    if self.stat:
                        self.stat.on_step(action, reward, terminal, q, v, a, l)
                if terminal:
                    break
        if monitor:
            self.env.monitor.close()

    def predict(self, state, is_train):
        u = self.pred_network.predict([state])[0]
        if is_train:
            noise_scale = 1/(self.idx_episode + 1)
            return u + noise_scale*np.random.randn(self.action_size)
        else:
            return u

    def perceive(self):

        q_list = []
        v_list = []
        a_list = []
        l_list = []

        for iteration in range(self.update_repeat):
            if len(self.rewards) >= self.batch_size:
                indexes = np.random.choice(len(self.rewards), size=self.batch_size)
            else:
                indexes = np.arange(len(self.rewards))

            x_t = np.array(self.prestates)[indexes]
            x_t_plus_1 = np.array(self.poststates)[indexes]
            r_t = np.array(self.rewards)[indexes]
            u_t = np.array(self.actions)[indexes]

            v = self.target_network.predict_v(x_t_plus_1, u_t)
            target_y = self.discount * np.squeeze(v) + r_t

            _, l, q, v, a = self.sess.run([
                self.optim, self.loss,
                self.pred_network.Q, self.pred_network.V, self.pred_network.A,
            ], {
                self.target_y: target_y,
                self.pred_network.x: x_t,
                self.pred_network.u: u_t,
                self.pred_network.is_train: True,
            })

            q_list.extend(q)
            v_list.extend(v)
            a_list.extend(a)
            l_list.append(l)

            self.target_network.soft_update_from(self.pred_network)

            logger.debug("q: %s, v: %s, a: %s, l: %s" \
                         % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))
        self.losses.append(np.mean(l))
        self.vs.append(np.mean(v))
        return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)
