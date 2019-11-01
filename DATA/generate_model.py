# Regression problem
# A surrogate model

from __future__ import absolute_import, division, print_function, unicode_literals

import pickle

import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import dill

print(tf.__version__)

data_dir = 'modelData/'
# Load the generated data

dataset = pd.read_hdf('data_all').dropna(axis=1)

# check
print('check:', dataset.tail())
# check
print('check nans:', dataset.isna().sum())

# Now split the dataset into a training set and a test set

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Overview
# sns.pairplot(train_dataset, diag_kind="kde")
# plt.show()
labels = [col for col in train_dataset.columns if 'BPM' in col or 'BPUSE' in col]
actors = [col for col in train_dataset.columns if not('BPM' in col or 'BPUSE' in col)]


train_stats = train_dataset.describe().transpose()
print(train_stats.columns)
train_stats_in = train_stats.loc[actors ,:]
print('train stats actors:\n', train_stats_in)

# Split labels...

train_labels = train_dataset[labels]
test_labels = test_dataset[labels]

def norm(x):
    return (x - train_stats_in['min']) / train_stats_in['max']

normed_train_data = norm(train_dataset).dropna(axis=1)
normed_test_data = norm(test_dataset).dropna(axis=1)

train_labels_stats = train_stats.loc[labels, :]
print('train stats labels:\n', train_labels_stats)

def norm_labes(x):
    return (x - train_labels_stats['min']) / train_labels_stats['max']

normed_train_labels_data = norm_labes(train_labels).dropna(axis=1)
normed_test_labels_data = norm_labes(test_labels).dropna(axis=1)

print('normed_test_data:\n', normed_test_labels_data)

# Create model:

def build_model():
    model = keras.Sequential([
        layers.Dense(400, activation=tf.nn.relu, input_shape=[len(normed_train_data.keys())]),
        layers.Dense(300, activation=tf.nn.relu),
        layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()

model.summary()

# Test of the model
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0,20])
    plt.legend()
    plt.show()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0: print('')
        print('.', end='')

EPOCHS = 100

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

history = model.fit(normed_train_data, normed_train_labels_data, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, normed_test_labels_data, verbose=0)

print("\nTesting set Mean Abs Error: {:5.5f}".format(mae))

test_predictions = model.predict(normed_test_data)

data_corr = normed_test_labels_data.copy()
data_corr['predict emitX'] = test_predictions[:, 0]
data_corr['predict pz_spread'] = test_predictions[:, 1]
sns.pairplot(data_corr, diag_kind="kde")
plt.show()

test_predictions = model.predict(normed_test_data)
error = test_predictions - test_labels
error.hist(bins=25)
plt.show()


def save_model(name, data_dir):
    filepath = data_dir + name + '_model'
    tf.keras.models.save_model(
        model,
        filepath,
        overwrite=True,
        include_optimizer=True,
        # save_format=None
    )
    filepath = data_dir + name + '_train_stats.dll'
    with open(filepath, 'wb') as output:
        dill.dump(train_dataset.describe(), output)


save_model('test', data_dir)
