import os
import pprint
import tensorflow as tf

from NAF_online.src.network import *

pp = pprint.PrettyPrinter().pprint

def get_model_dir(config, exceptions=None):

  attrs = config.__flags
  pp(attrs)

  keys = attrs.keys()

  names =[]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'

def preprocess_conf(conf):
  options = conf.__flags
  for option, value in options.items():
    option = option.lower()
    value = value.value
    if option == 'hidden_dims':
      conf.hidden_dims = eval(conf.hidden_dims)
    elif option.endswith('_w'):
      weights_initializer = tf.random_uniform_initializer(-0.05, 0.05)
      setattr(conf, option, weights_initializer)
    elif option.endswith('_fn'):
      activation_fn = tf.nn.tanh
      setattr(conf, option, activation_fn)
