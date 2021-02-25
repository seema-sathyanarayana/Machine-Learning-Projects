#!/usr/bin/env python
# coding: utf-8

# ### Importing the Packages

# In[1]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the matplotlib, seaborn, numpy and pandas packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the clustering related packages
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# ### Reading the CSV file

# In[2]:


country_df = pd.read_csv('Country-data.csv')
country_df.head()


# ### Understanding the data

# ##### Shape of dataframe

# In[3]:


country_df.shape


# ##### Information on dataframe

# In[4]:


country_df.info()


# ##### Null Percentage

# In[5]:


round(100*(country_df.isnull().sum())/len(country_df), 2)


# **Observation** There are no null values in the dataframe

# ##### Converting columns exports, import and health which is in percentage

# In[6]:


country_df['export_nos'] = round(country_df.exports/100 * country_df.gdpp, 2)
country_df['import_nos'] = round(country_df.imports/100 * country_df.gdpp, 2)
country_df['health_nos'] = round(country_df.health/100 * country_df.gdpp, 2)
country_df.head()


# ##### Description of numerical variables of modified dataframe

# In[7]:


country_df.describe()


# ### Univariate Analysis

# In[8]:


sns.boxplot(country_df.child_mort)
plt.show()


# **Observation** There are few countries which have very child mortality rate

# In[9]:


sns.boxplot(country_df.health_nos)
plt.show()


# In[10]:


sns.boxplot(country_df.income)
plt.show()


# In[11]:


sns.boxplot(country_df.gdpp)
plt.show()


# ### Bivariate Analysis

# In[12]:


sns.pairplot(country_df[['export_nos','import_nos','health_nos','child_mort','income','gdpp']])


# In[13]:


plt.figure(figsize=(50,10))
sns.barplot(country_df.country, country_df.gdpp)
plt.xticks(rotation=90)
plt.show()


# In[14]:


plt.figure(figsize=(50,10))
sns.barplot(country_df.country, country_df.child_mort)
plt.xticks(rotation=90)
plt.show()


# In[15]:


plt.figure(figsize=(50,10))
sns.barplot(country_df.country, country_df.income)
plt.xticks(rotation=90)
plt.show()


# ### Outlier Treatment

# - After capping the column health_nos to 1% and 99%, there is no much difference seen

# In[16]:


country_df.health_nos = np.clip(country_df.health_nos, country_df.health_nos.quantile([0.01, 0.99][0]),
                          country_df.health_nos.quantile([0.01, 0.99][1]))
sns.boxplot(country_df.health_nos)
plt.show()


# - After capping the column Child Mortality to 1% and 99%

# In[17]:


country_df.child_mort = np.clip(country_df.child_mort, country_df.child_mort.quantile([0.01, 0.99][0]),
                          country_df.child_mort.quantile([0.01, 0.99][1]))
sns.boxplot(country_df.child_mort)
plt.show()


# - After capping the column income to 1% and 99%

# In[18]:


country_df.income = np.clip(country_df.income, country_df.income.quantile([0.01, 0.99][0]),
                          country_df.income.quantile([0.01, 0.99][1]))
sns.boxplot(country_df.income)
plt.show()


# - After capping the column gdpp to 1% and 99%

# In[19]:


country_df.gdpp = np.clip(country_df.gdpp, country_df.gdpp.quantile([0.01, 0.99][0]),
                          country_df.gdpp.quantile([0.01, 0.99][1]))
sns.boxplot(country_df.gdpp)
plt.show()


# ### Scaling

# In[20]:


group_df = country_df[['country', 'child_mort', 'gdpp', 'income']]

# instantiate
scaler = StandardScaler()

# fit_transform
group_df_scaled = scaler.fit_transform(group_df[['child_mort', 'gdpp', 'income']])
group_df_scaled.shape


# In[21]:


group_df_scaled = pd.DataFrame(group_df_scaled)
group_df_scaled.columns = ['child_mort', 'gdpp', 'income']
group_df_scaled.head()


# ### Hopkins Statistics

# In[22]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# In[23]:


hopkins(group_df_scaled)


# - The value `0.95` is `1` implies that the data can be clustered.

# ### K-Means Clustering

# ##### k-means with some arbitrary k (k = 4)

# In[24]:


kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(group_df_scaled)


# In[25]:


kmeans.labels_


# ##### Elbow curve

# In[26]:


ssd = []
range_n_clusters = [2, 3, 4, 5, 6]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(group_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSD for each n clusters
plt.plot(ssd)
plt.xlabel('k values 2-6')
plt.ylabel('SSD')
plt.show()


# ##### Silhouette Analysis

# In[27]:


range_n_clusters = [2, 3, 4, 5, 6]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(group_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(group_df_scaled, cluster_labels)
    print("For n clusters = {0}, the silhouette score : {1}".format(num_clusters, silhouette_avg))


# - As the silhouette score of n cluster 3 and 4 are almost same that is 0.54, picking K to be 3. Even there is a significant bent in the elbow for K=3

# ##### Final model using K-Means with k = 3

# In[28]:


kmeans = KMeans(n_clusters=3, max_iter=50, random_state=50)
kmeans.fit(group_df_scaled)


# In[29]:


kmeans.labels_


# In[30]:


kmeans_df = pd.DataFrame(kmeans.labels_, columns=['cluster_id'])
kmeans_df = pd.concat([group_df, kmeans_df], axis=1)
kmeans_df.head()


# ##### Child mortality rate with respect clusters obtained

# In[31]:


sns.boxplot(kmeans_df.cluster_id, kmeans_df.child_mort)
plt.show()


# - From above graph, we can say that `cluster 1` has high child mortality rate and `cluster 0` has less child mortality rate

# ##### Income range with respect clusters obtained

# In[32]:


sns.boxplot(kmeans_df.cluster_id, kmeans_df.income)
plt.show()


# - From above graph, we can say that `cluster 1` has low income and `cluster 0` has high income.

# ##### GDPP with respect clusters obtained

# In[33]:


sns.boxplot(kmeans_df.cluster_id, kmeans_df.gdpp)
plt.show()


# - From above graph, we can say that `cluster 1` has low GDPP and `cluster 0` has high GDPP

# In[34]:


kmeans_df.groupby(by='cluster_id').mean()


# In[35]:


kmeans_df.groupby(by='cluster_id').mean().plot(kind='bar')
plt.show()


# - Based on the above graphs and table, `cluster 1` is the intented group which is in need of aid as the countries in this group has `low GDPP, low Income and high Child mortality rate`.

# ##### Dataframe of countries with low GDPP, low Income and high Child mortality rate

# In[36]:


under_develop = kmeans_df[kmeans_df.cluster_id == 1]
under_develop.head()


# ##### Top 5 countries with highest child mortality rate

# In[37]:


under_develop.sort_values(by='child_mort', ascending=False).head()


# ##### Top 5 countries with low income

# In[38]:


under_develop.sort_values(by='income').head()


# ##### Top 5 countries with low gdpp

# In[39]:


under_develop.sort_values(by='gdpp').head()


# - GDPP gives an overall view of the country's socio-economic status and also its contribution for health sector.

# ### Hierarchical Clustering

# In[40]:


group_df_scaled.head()


# In[41]:


group_df.head()


# ##### Single Linkage

# In[42]:


plt.figure(figsize=(17,5))
merging = linkage(group_df_scaled, method='single', metric='euclidean')
dendrogram(merging)
plt.show()


# ##### Complete Linkage

# In[43]:


plt.figure(figsize=(17,5))
merging = linkage(group_df_scaled, method='complete', metric='euclidean')
dendrogram(merging)
plt.show()


# ##### Selecting n_clusters to be 3

# In[44]:


cluster_labels = cut_tree(merging, n_clusters=3).reshape(-1,)
cluster_labels


# In[45]:


hierarchical_df = pd.DataFrame(cluster_labels, columns=['cluster_labels'])
hierarchical_df = pd.concat([group_df, hierarchical_df], axis=1)
hierarchical_df.head()


# ##### Child mortality rate with respect clusters obtained

# In[46]:


sns.boxplot(hierarchical_df.cluster_labels, hierarchical_df.child_mort)
plt.show()


# - From above graph, we can say that `cluster 0` has high child mortality rate and `cluster 2` has less child mortality rate

# ##### Income range with respect clusters obtained

# In[47]:


sns.boxplot(hierarchical_df.cluster_labels, hierarchical_df.income)
plt.show()


# - From above graph, we can say that `cluster 0` has low income and `cluster 2` has high income.

# ##### GDPP with respect clusters obtained

# In[48]:


sns.boxplot(hierarchical_df.cluster_labels, hierarchical_df.gdpp)
plt.show()


# - From above graph, we can say that `cluster 0` has low gdpp and `cluster 2` has high gdpp.

# In[49]:


hierarchical_df.groupby(by='cluster_labels').mean()


# In[50]:


hierarchical_df.groupby(by='cluster_labels').mean().plot(kind='bar')
plt.show()


# In[51]:


under_developed = hierarchical_df[hierarchical_df.cluster_labels == 0]
under_developed.head()


# In[52]:


under_developed.sort_values(by='child_mort', ascending=False).head()


# In[53]:


under_developed.sort_values(by='income').head()


# In[54]:


under_developed.sort_values(by='gdpp').head()


# - GDPP gives an overall view of the country's socio-economic status and also its contribution for health sector.

# The final list of 5 countries which are in direst need of aid are
# 1. Liberia
# 2. Burundi
# 3. Congo, Dem. Rep.
# 4. Niger
# 5. Sierra Leone

# In[ ]:




