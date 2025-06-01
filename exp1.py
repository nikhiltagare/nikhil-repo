import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import fetch_california_housing 
california_housing = fetch_california_housing() 
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names) 
numerical_features = data.select_dtypes(include=[np.number]).columns 
print(numerical_features) 
data.hist(bins=30, figsize=(12, 7),color='blue') 
plt.suptitle('Histograms of Numerical Features') 
plt.tight_layout() 
plt.savefig('exp1-1.png')
plt.show() 
plt.figure(figsize=(15, 10)) 
for i, column in enumerate(data.columns, 1): 
 plt.subplot(3, 3, i) 
 sns.boxplot(y=data[column]) 
 plt.title(f'Box Plot of {column}') 
plt.tight_layout() 
plt.savefig('exp1-2.png')
plt.show() 
print("Outliers Detection:\n") 
outliers_summary = {} 
for feature in numerical_features: 
 Q1 = data[feature].quantile(0.25) 
 Q3 = data[feature].quantile(0.75) 
 IQR = Q3 - Q1 
 lower_bound = Q1 - 1.5 * IQR 
 upper_bound = Q3 + 1.5 * IQR 
 outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)] 
 outliers_summary[feature] = len(outliers) 
 print(f"\t{feature}: {len(outliers)} outliers\t")