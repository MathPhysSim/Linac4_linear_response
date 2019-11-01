import os
#
import pandas as pd
import matplotlib.pyplot as plt

import h5py
my_file = 'init_states.h5'
frame = pd.read_hdf(my_file, key='init')

print(frame.loc[:,[col for col in frame.columns if ('BCT' or 'BPUSE' in col) ]].T.plot())
plt.show()


print(frame.loc[:,[col for col in frame.columns if not('BCT' or 'BPUSE' in col) ]].T.plot())
plt.show()