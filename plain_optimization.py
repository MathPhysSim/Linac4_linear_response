# import pybobyqa
from bayes_opt import BayesianOptimization

from simple_environment_linac4 import simpleEnv
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
import time

# import PyQt5
# matplotlib.use("Qt5Agg")



class EnvironmentWrapper():
    def __init__(self, **kwargs):
        self.env = simpleEnv(**kwargs)
        self.env.reset()
        self.action_space_dimensions = self.env.action_space.shape[0]
        self.action_bounds_high, self.action_bounds_low = self.env.action_space.high, self.env.action_space.low
        # print(self.action_bounds_high, self.action_bounds_low)
        self.action_names = range(self.env.action_space.shape[0])
        self.history = pd.DataFrame(columns=self.action_names)
        self.state_history_mean = pd.DataFrame()
        self.state_history_std = pd.DataFrame()

        self.counter = -1
        self.session_name = 1
        self.algorithm_name = None

    def set_history(self, action, r, s):
        self.counter += 1
        size = self.history.shape[0]
        self.history.loc[size, self.action_names] = action
        self.history.loc[size, 'objective'] = r


    def plot_optimization(self):
        self.ax1.set_title('Actors')
        self.ax2.set_title('Objective')
        self.history.loc[:, self.action_names].plot(ax=self.ax1, legend=False)
        self.history.loc[:, 'objective'].plot(ax=self.ax2)

    def objective(self, action):
        s, r, d, _ = self.env.step(action=action)
        self.set_history(action, r=r, s=s)
        return -r


algorithm_list = ['BAYESIAN', 'BOBYQA', 'Powell', 'COBYLA']
algotihm_name = algorithm_list[-1]

if __name__ == '__main__':
    print('starting the algorithm:', algotihm_name)


    environment_instance = EnvironmentWrapper()

    start_vector = np.zeros(environment_instance.action_space_dimensions)

    if algotihm_name == 'COBYLA':
        def constr(action):
            if any(action > environment_instance.action_bounds_high):
                return -1
            elif any(action < environment_instance.action_bounds_low):
                return -1
            else:
                return 1
        rhobeg = 0.5 * environment_instance.action_bounds_high[0]
        solution = opt.fmin_cobyla(environment_instance.objective, start_vector, [constr], rhobeg=rhobeg, rhoend=.01)

    elif algotihm_name == 'BOBYQA':
        lower = environment_instance.action_bounds_low
        upper = environment_instance.action_bounds_high
        rhobeg = 0.75 * environment_instance.action_bounds_high[0]
        res = pybobyqa.solve(environment_instance.objective, start_vector, seek_global_minimum=False, rhobeg=rhobeg,
                             objfun_has_noise=True, bounds=(lower, upper))

    elif algotihm_name == 'Powell':
        res = opt.fmin_powell(environment_instance.objective, start_vector, ftol=0.1, xtol=0.1,
                              direc=0.5 * environment_instance.action_bounds_high[0])

    print(environment_instance.env.step(solution))
    environment_instance.fig = plt.figure()
    environment_instance.ax1 = environment_instance.fig.add_subplot(211)
    environment_instance.ax2 = environment_instance.fig.add_subplot(212, sharex=environment_instance.ax1)
    environment_instance.plot_optimization()
    plt.show()
