#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
get_ipython().run_line_magic('matplotlib', 'inline')

boston = load_boston()

bostonDF=pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF['PRICE']=boston.target


# In[14]:



print('boston 데이터 세트 : ', bostonDF.shape)
bostonDF.head()


# In[18]:


fig, axs = plt.subplots(figsize=(16,8), ncols=4, nrows=2)
lm_features = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD']
for i, feature in enumerate(lm_features):
    row = int(i/4)
    col = i%4
    sns.regplot(x=feature, y='PRICE', data=bostonDF, ax=axs[row][col])


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = bostonDF['PRICE']
X_data = bostonDF.drop(['PRICE'], axis=1, inplace=False)

X_train, X_test, y_train, y_test=train_test_split(X_data, y_target, test_size=0.3, random_state=156)

lr=LinearRegression()
lr.fit(X_train, y_train)
y_preds=lr.predict(X_test)
mse=mean_squared_error(y_test, y_preds)
rmse= np.sqrt(mse)

print('MSE : {:.3f}, RMSE : {:.3f}'.format(mse, rmse))
print('Variance score : {:.3f} '.format(r2_score(y_test, y_preds)))


# In[27]:


print('절편값 : {},\n 회기계수값 : {}'.format(lr.intercept_, np.round(lr.coef_,1)))


# In[29]:


coeff=pd.Series(data=np.round(lr.coef_,1), index=X_data.columns)
coeff.sort_values(ascending=False)


# In[37]:


from sklearn.preprocessing import PolynomialFeatures

X=np.arange(4).reshape(2,2)
print(X)
X[:,0]


# In[34]:


poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr=poly.transform(X)
print(poly_ftr)


# In[ ]:




