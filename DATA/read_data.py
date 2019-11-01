import os
#
import pandas as pd

import h5py

directory = 'DATA/'
file_array = [file for file in os.listdir(directory)]
data_all = pd.DataFrame()

for my_file in file_array:
    my_file = directory + my_file
    try:
        with h5py.File(my_file) as f:
            keys_array = [key for key in f.keys() if 'default' in key]
        for k in keys_array:
            k = k + '/measured' if 'default' in k else k
            frame = pd.read_hdf(my_file, key=k)
            print(frame)
            data_all = pd.concat([data_all, frame], ignore_index=True)

        print('loaded data:', my_file)
    except:
        print('failed at:', my_file)

print(data_all.loc[:,[col for col in data_all.columns if not('BCT' in col) ]])
data_all = data_all.loc[:,[col for col in data_all.columns if not('BCT' in col) ]]

data_all.to_hdf('data_all', key='data')

# data_all = pd.read_hdf('data_all')
# data_all = data_all.dropna()

