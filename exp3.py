import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data 
y = iris.target 
label_names = iris.target_names
df = pd.DataFrame(X, columns=iris.feature_names)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
df_pca = pd.DataFrame(data=principal_components,columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y
plt.figure(figsize=(6, 4))
colors = ['y', 'b', 'g']
for i, label in enumerate(np.unique(y)):
    plt.scatter(df_pca[df_pca['Target'] == label]['Principal Component 1'],df_pca[df_pca['Target'] == label]['Principal Component 2'],label=label_names[label],
color=colors[i])
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
print('The output is saved as pcaofirisdataset.png')
plt.savefig('exp3-1.png')
plt.show()
