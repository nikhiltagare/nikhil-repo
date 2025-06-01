import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

corr_matrix = data.corr()
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap= 'crest', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.savefig('exp2-1.png')
plt.show()

plt.figure()
sns.pairplot(data, kind='scatter',diag_kind='kde', plot_kws={'alpha': 0.5})
plt.savefig('exp2-2.png')
plt.show()
