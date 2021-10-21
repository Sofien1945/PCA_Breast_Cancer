"""Breast cancer case study using Principal component analysis
Part of Simplearn machine learning course
Date: 21.10.2021
Done By: Sofien Abidi"""

#Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#Import Cancer Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())

#Convert Dataset to dataframe
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
#df['target_names'] = cancer.target
print(df.shape)

#Data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
#print(scaled_data)

#Call PCA function
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

#Plt PCA result
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c = cancer["target"], cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

df_comp = pd.DataFrame(pca.components_, columns = cancer.feature_names)
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')
plt.show()
