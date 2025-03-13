#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("ecommerce.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


#EDA

sns.jointplot(x="Time on Website",y="Yearly Amount Spent",data=df,alpha=0.5)


# In[7]:


sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=df,alpha=0.5)


# In[8]:


sns.pairplot(df,kind='scatter',plot_kws={'alpha':0.4})


# In[9]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data = df,scatter_kws={'alpha':0.3})


# In[10]:


from sklearn.model_selection import train_test_split


# In[15]:


X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']


# In[16]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)


# In[ ]:


#Training the model


# In[20]:


from sklearn.linear_model import LinearRegression


# In[22]:


model = LinearRegression()


# In[23]:


model.fit(X_train,y_train)


# In[24]:


model.coef_


# In[25]:


cdf = pd.DataFrame(model.coef_,X.columns,columns=['Coef'])
cdf


# In[ ]:


#Predictions


# In[26]:


predictions = model.predict(X_test)
predictions


# In[32]:


sns.scatterplot(x=predictions,y=y_test)
plt.xlabel("Predictions")
plt.title("Evaluation of our Linear Regression model")


# In[36]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
import math


# In[37]:


print("Mean Absolute Error: ",mean_absolute_error(y_test,predictions))
print("Mean Squared Error: ",mean_squared_error(y_test,predictions))
print("RMSE: ",math.sqrt(mean_squared_error(y_test,predictions)))


# In[ ]:


#Residuals


# In[38]:


residuals = y_test - predictions


# In[39]:


residuals


# In[43]:


sns.distplot(residuals,bins=20,kde=True)


# In[44]:


import pylab
import scipy.stats as stats

stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()


# In[ ]:




