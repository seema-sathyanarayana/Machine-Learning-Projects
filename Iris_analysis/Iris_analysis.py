
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cross_validation  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris_data = pd.read_csv('Iris.csv') #reading the csv file
iris_data.head() #printing first 5 rows


# In[3]:


#converting dtabular data to html page
report = pandas_profiling.ProfileReport(iris_data)
report.to_file("iris_data.html")


# In[4]:


iris_data.corr() #correlation columnwise


# In[5]:


sns.pairplot(iris_data, vars=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], hue='Species')
# this plot gives relation between every variable in data to other.
# we can say that PetalLengthCm and PetalWidthCm are highly correlated to each other.


# In[6]:


plt.rcParams['figure.figsize'] = (15, 6)
sns.barplot(y='SepalLengthCm', x='SepalWidthCm', data=iris_data, hue='Species')
# vericolor has smaller sepal width compared to setosa and virginica has moderate sepal width.
# on an average sepal length of virginica is more compared to other species.


# In[7]:


sns.pointplot(y='SepalLengthCm', x='PetalLengthCm', data=iris_data, hue='Species')
# petal length is more for virginica when compared to other species.
# setosa has least petal length among the three species.


# In[8]:


sns.pointplot(y='SepalLengthCm', x='PetalWidthCm', data=iris_data, hue='Species')
# petal width is more for virginica when compared to other species.
# setosa has least petal width among the three species.
# from all the above 2 graph including this, we can say that Sepal Length of virginica is more compared to others and setosa has
# less Sepal length on comparison, whatever may be the independent variable.


# In[9]:


sns.boxplot(y='SepalLengthCm', x='Species', data=iris_data)
# we can say that Sepal Length of virginica is more compared to others and setosa has less Sepal length on comparison
# virginica has more spread i.e., variance is more.


# In[10]:


sns.boxplot(y='SepalWidthCm', x='Species', data=iris_data)
# setosa has more spread i.e., data ranges from 2.3 to 4.4 around
# sepal width of setosa is more.


# In[11]:


sns.boxplot(y='PetalLengthCm', x='Species', data=iris_data)
# setosa has very less spread i.e., data is more concentrated in a region. petal length of setosa is very less.
# virginica has larger petal length.


# In[12]:


sns.boxplot(y='PetalWidthCm', x='Species', data=iris_data)
# setosa has very less spread i.e., data is more concentrated in a region. petal width of setosa is very less.
# virginica has larger petal width.
# from the box plots we can say that versicolor is moderately shaped species of flower.


# In[13]:


iris_data.groupby('Species').mean()


# In[14]:


# Seaborn scatter plot with regression line
sns.lmplot(x='PetalLengthCm', y='SepalLengthCm', data=iris_data, aspect=1.0, scatter_kws={'alpha':0.4})


# In[15]:


feature_cols = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
sns.pairplot(iris_data, x_vars=feature_cols, y_vars='SepalLengthCm', kind='reg', size=3.5, aspect=1)
# we can see from the below graph that Sepal Length and Petal Length is linearly related.
# even Sepal Length and Petal Width is also linearly related.


# In[16]:


# create X and y
X1 = iris_data[['PetalLengthCm']]
X2 = iris_data[['PetalWidthCm']]
X3 = iris_data[['SepalWidthCm']]
y = iris_data.SepalLengthCm


# In[17]:


# import, instantiate, fit
linreg1 = LinearRegression()
linreg1.fit(X1, y)


# In[18]:


# print the coefficients
print (linreg1.intercept_)
print (linreg1.coef_)


# In[19]:


# y = mx + c
print(linreg1.intercept_ + linreg1.coef_*2)
# use the predict method
linreg1.predict(2)


# In[20]:


linreg2 = LinearRegression()
linreg2.fit(X2, y)
print (linreg2.intercept_)
print (linreg2.coef_)
print(linreg2.intercept_ + linreg2.coef_*3)
linreg2.predict(3)


# In[21]:


linreg3 = LinearRegression()
linreg3.fit(X3, y)
print (linreg3.intercept_)
print (linreg3.coef_)
print(linreg3.intercept_ + linreg3.coef_*5)
linreg3.predict(5)


# In[25]:


# pair the feature names with the coefficients
X = iris_data[feature_cols]
y = iris_data.SepalLengthCm
linreg = LinearRegression()
linreg.fit(X,y)
list(zip(feature_cols, linreg.coef_))


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **SepalWidthCm** is associated with a **SepalLengthCm increase of 0.65cm**.
# - Holding all other features fixed, a 1 unit increase in **PetalLengthCm** is associated with a **SepalLengthCm increase of 0.711cm**.
# - Holding all other features fixed, a 1 unit increase in **PetalWidthCm** is associated with a **SepalLengthCm decreases of -0.56cm**.
