import gym
import tensorflow as tf

from spinup.algos.ddpg.ddpg import ddpg
from spinup.utils.run_utils import ExperimentGrid

env_fn = lambda : gym.make('Pendulum-v0')

network_kwargs = dict(hidden_sizes=[400, 300], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='logging/NAF', exp_name='naf - tests')

steps_per_epoch = 1000
epochs = 100
start_steps = 50
algorithm = 'naf'

if __name__ == '__main__':

    eg = ExperimentGrid(name='ddpg-bench-long')
    eg.add('env_name', 'Pendulum-v0', '', True)
    eg.add('seed', [10 * i for i in range(4)])
    eg.add('epochs', 20)
    eg.add('steps_per_epoch', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(100, 100), (400, 300)], 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.run(ddpg, num_cpu=4, data_dir='logging/DDPG')
#
# # agent = naf(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=100, epochs=25, logger_kwargs=logger_kwargs)
#
# # agent = spinup.ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=500, epochs=250, logger_kwargs=logger_kwargs,
# #                     start_steps=start_steps)

# tf.reset_default_graph()
# naf(env_fn=env_fn, ac_kwargs=network_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs, logger_kwargs=logger_kwargs,
#     act_noise=0.1, start_steps=start_steps)