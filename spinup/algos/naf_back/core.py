import numpy as np
import tensorflow as tf


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]


def mpl_inner(x, num_outputs, activation_fn, weights_initializer, biases_initializer, scope):
    return tf.contrib.layers.fully_connected(x, num_outputs=num_outputs, activation_fn=activation_fn,
                                                 weights_initializer=weights_initializer,
                                                 biases_initializer=biases_initializer, scope=scope)


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=tf.tanh, weight_init=None, scope='mpl'):
    for idx, h in enumerate(hidden_sizes[:-1]):
        x = mpl_inner(x, num_outputs=int(h), activation_fn=activation,weights_initializer=weight_init,
                      biases_initializer=tf.constant_initializer(0.0), scope=scope + '%d' % idx)
    if len(hidden_sizes) > 1:
        idx += 1
        scope = scope + '%d' % idx
    return mpl_inner(x, num_outputs=int(hidden_sizes[-1]), activation_fn=output_activation,
                     weights_initializer=weight_init,
                     biases_initializer=tf.constant_initializer(0.0), scope=scope)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


"""
Normalize Advantage Function
"""


def mlp_normalized_advantage_function(x, a, hidden_sizes=(100, 100), activation=tf.tanh,
                                      output_activation=tf.tanh, action_space=None, weight_init=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]

    # create a shared network for the variables
    hid_outs = {}
    # TODO: Check the availability of the variables in the scope
    with tf.name_scope('hidden'):
        h = mlp(x, list(hidden_sizes), activation, output_activation, weight_init, scope='hid')
        hid_outs['v'], hid_outs['l'], hid_outs['mu'] = h, h, h

    with tf.name_scope('value'):
        V = mlp(hid_outs['v'], [1], None, None, weight_init, scope='V')

    with tf.name_scope('advantage'):
        l = mlp(hid_outs['l'], [(act_dim * (act_dim + 1)) / 2], None, None, weight_init, scope='l')
        mu = act_limit * mlp(hid_outs['mu'], [act_dim], activation, output_activation, weight_init, scope='mu')
        pivot = 0
        rows = []
        for idx in range(act_dim):
            count = act_dim - idx
            diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
            non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
            row = tf.pad(tf.concat((diag_elem, non_diag_elems), 1), ((0, 0), (idx, 0)))
            rows.append(row)
            pivot += count

        L = tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))
        P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))
        tmp = tf.expand_dims(a - mu, -1)
        A = -tf.matmul(tf.transpose(tmp, [0, 2, 1]), tf.matmul(P, tmp)) / 2
        A = tf.reshape(A, [-1, 1])

    with tf.name_scope('Q'):
        Q = A + V

    return mu, V, Q, P, A
