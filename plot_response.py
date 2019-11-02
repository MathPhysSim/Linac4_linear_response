import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle('linac4_rm.pcl')
df.columns = [entry.split('/')[0] for entry in df.columns]
df.index = [entry.split('/')[0] for entry in df.index]
# sns.pairplot(df.iloc[:16,:16], size=2.0)
# # plt.show()
df = df.iloc[17:, 16:]
# df = df.iloc[17:, 16:]
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)
hm = sns.heatmap(df,
                 # cbar=True,
                 # annot=True,
                 square=True,
                 # fmt='.2f',
                 # annot_kws={'size': 10},
                 cmap="YlGnBu")
plt.title('Vertical response', size = 18)
plt.tight_layout()
plt.savefig('Response_matrix_ver.png')
plt.show()
