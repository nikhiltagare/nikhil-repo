from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np1
import numpy.linalg as np
from scipy.stats import pearsonr

def kernel(point,xmat, k):
    m,n = np1.shape(xmat)
    weights = np1.mat(np1.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np1.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    #W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
    XT_WX = X.T * (wei * X)
    XT_WX += np1.eye(XT_WX.shape[0]) * 1e-5
    W = XT_WX.I * (X.T * (wei * ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    m,n = np1.shape(xmat)
    ypred = np1.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

data = pd.read_csv('tips.csv')
bill = np1.array(data.total_bill)
tip = np1.array(data.tip)
mbill = np1.mat(bill)
mtip = np1.mat(tip)
m= np1.shape(mbill)[1]
one = np1.mat(np1.ones(m))
X= np1.hstack((one.T,mbill.T))
# print(X)


ypred = localWeightRegression(X,mtip,0.3)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='blue', label = 'Actual Data')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=2, label='Predicted Regression Line')
plt.xlabel('Total bill', fontsize = 14)
plt.ylabel('Tip', fontsize = 14)

plt.title('Locally Weighted Regression', fontsize=18)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('exp6-1.png')
plt.show();
