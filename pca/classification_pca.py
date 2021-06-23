import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt # NOTE: This was tested with matplotlib v. 2.1.0

datasets = pd.read_csv('data_cancer.csv')
dataset = datasets.drop(columns=['Unnamed: 32'])
datasets.head()
datasets.describe()
datasets.corr()
fig=plt.subplots(figsize=(15,15))
corr=dataset.corr()
sns.heatmap(corr, annot=True)
plt.show()
