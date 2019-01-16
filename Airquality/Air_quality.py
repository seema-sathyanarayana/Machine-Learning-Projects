
# coding: utf-8

# ### **Domain Name :** Environment air quality 
# 
# **Dataset :** Air quality of an Italian city
# 
# *   The dataset contains 9358 instances of hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. 
# 
# *   The device was located on the field in a significantly polluted area, at road level, within an Italian city. 
# 
# *   Data were recorded from March 2004 to February 2005 (one year)representing the longest freely available recordings of on field deployed air quality chemical sensor devices responses.
# 
# *   Ground Truth hourly averaged concentrations for CO, Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) and were provided by a co-located reference certified analyzer. 
# 
# *   Missing values are tagged with -200 value. 
# 
# **Attributes of the dataset are :**
# 0.	 Date (DD/MM/YYYY) 
# 1.	Time (HH.MM.SS)
# 2.	True hourly averaged concentration CO in mg/m^3 (reference analyzer)
# 3.	PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
# 4.	True hourly averaged overall Non MetanicHydroCarbons concentration in microg/m^3 (reference analyzer)
# 5.	True hourly averaged Benzene concentration in microg/m^3 (reference analyzer) 
# 6.	PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted) 
# 7.	True hourly averaged NOx concentration in ppb (reference analyzer) 
# 8.	PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
# 9.	True hourly averaged NO2 concentration in microg/m^3 (reference analyzer) 
# 10.	PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted) 
# 11.	PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
# 12.	Temperature in °C 
# 13.	Relative Humidity (%) 
# 14.	AH Absolute Humidity
# 
# **Problem :**
# 
# Humans are very sensitive to humidity, as the skin relies on the air to get rid of moisture. The process of sweating is your body's attempt to keep cool and maintain its current temperature. If the air is at 100 % relative humidity, sweat will not evaporate into the air. As a result, we feel much hotter than the actual temperature when the relative humidity is high. If the relative humidity is low, we can feel much cooler than the actual temperature because our sweat evaporates easily, cooling ­us off. For example, if the air temperature e is 75 degrees Fahrenheit (24 degrees Celsius) and the relative humidity is zero percent, the air temperature feels like 69 degrees Fahrenheit (21 C) to our bodies. If the air temperature is 75 degrees Fahrenheit (24 C) and the relative humidity is 100 percent, we feel like it's 80 degrees (27 C) out. 
# 
# **Objective :**
# 
# The objective is to predict the Relative Humidity at a given point of time based on all other attributes affecting the change in RH.
# 
# **Humidity Concepts :**
# 
# * Absolute humidity is the water content of air. The hotter the air is, the more water it can contain implies higher the absolute humidity.
# * Relative humidity, expressed as a percent, measures the current absolute humidity relative to the maximum for that temperature (which depends on the current air temperature). when the moisture content remains constant and temperature increases, relative humidity decreases and vice versa.

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import datetime

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cluster import KMeans


# ### **Reading the Data**

# In[2]:


#dataset = pd.read_csv('https://raw.githubusercontent.com/seema-sathyanarayana/Machine-Learning-Projects/master/Airquality/AirQualityUCI.csv')
dataset = pd.read_csv('AirQualityUCI.csv')
print(dataset.head())


# ### **Knowing more about whether categorical or numerical and missing values**

# In[3]:


dataset.info()


# All the columns are numerical.
# Columns Unnamed: 15 and Unnamed: 16 has only NaN values, the column dropped as it doesn't add any value during analysis.

# In[4]:


dataset.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)  #Removing NaN values
dataset.drop(dataset.index[9357:9471], inplace=True)


# In[5]:


dataset.describe()


# In[6]:


dataset['Date_Time'] = dataset['Date'] + ' ' + dataset['Time']

def conv_date(date):
  req_date = '%m/%d/%Y %H:%M:%S' # The format
  datetime_obj = datetime.datetime.strptime(date, req_date)
  datetime.date.strftime(datetime_obj,'%d-%m-%y %H:%M')
  return datetime_obj

dataset['Date_Time'] = dataset.apply(lambda row: conv_date(row['Date_Time']), axis=1)


# In[7]:


new_data = dataset.drop(columns=['Date','Time'])
new_data = new_data.set_index('Date_Time')


# In[8]:


new_data.head()


# In[9]:


new_data.tail()


# In[10]:


plt.rcParams['figure.figsize'] = (200, 30)
sns.swarmplot(data=new_data, x='AH', y='RH', size=20)


# In[11]:


sns.swarmplot(data=new_data, x='T', y='RH', size=20)


# In[12]:


sns.swarmplot(data=new_data, x='PT08.S2(NMHC)', y='RH', size=20)


# In[13]:


sns.swarmplot(data=new_data, x='PT08.S1(CO)', y='RH', size=20)


# In[14]:


sns.swarmplot(data=new_data, x='C6H6(GT)', y='RH', size=20)


# In[15]:


sns.swarmplot(data=new_data, x='PT08.S5(O3)', y='RH', size=20)


# In[16]:


sns.swarmplot(data=new_data, x='PT08.S4(NO2)', y='RH', size=20)


# In[17]:


sns.swarmplot(data=new_data, x='NO2(GT)', y='RH', size=20)


# In[18]:


sns.swarmplot(data=new_data, x='PT08.S3(NOx)', y='RH', size=20)


# In[19]:


sns.swarmplot(data=new_data, x='NOx(GT)', y='RH', size=20)


# In[20]:


sns.swarmplot(data=new_data, x='NMHC(GT)', y='RH', size=20)


# In[21]:


sns.swarmplot(data=new_data, x='CO(GT)', y='RH', size=20)


# - RH ranges in between 9 to 90. Its almost constant throughout with respect to AH
# - For T below 2, RH is scarce whereas t RH is constant in range 10 to 90 for T between 2 to 21 later as T increases range of RH decreases to 10 to 20
# - RH is very less and scarce when PT08.S2(NMHC) conc is less than 300 and more than 500
# - RH is very less and scarce when PT08.S1(CO) conc is less than 300 and RH decreasing more than 500
# - RH is very less and scarce when C6H6(GT) conc increases above 23
# - RH is almost uniform for all PT08.S4(NO2), PT08.S3(NOx) and PT08.S5(O3) conc
# - RH is almost constant at lower conc of NO2(GT), NOx(GT) than at higher conc more than 300
# - RH is almost constant with few outliers for all NMHC(GT) conc
# - RH decreases above 5 for CO(GT) conc

# ### **Correlated Features**

# In[22]:


print(new_data.corr())
plt.rcParams['figure.figsize'] = (10, 10)
sns.heatmap(new_data.corr())


# From above matrix and heatmap, we can say that 
# 
# *  PT08.S1(CO) is highly correlated to PT08.S2(NMHC) and PT08.S5(O3)
# *  C6H6(GT) is highly corelated to T, RH and AH 
# *  PT08.S2(NMHC) is highly corelated to PT08.S5(O3) and PT08.S1(CO)
# *  T is highly correlated to AH and C6H6(GT)
# *  RH is highly correlated to AH and C6H6(GT)
# *  AH is highly correlated to T, RH and C6H6(GT)
# 
# The target Relative Humidity is highly correlated to AH and C6H6(GT)

# In[23]:


#new_data['DT_num'] = new_data.Date_Time.values.astype('float64')
#new_data.head()


# In[24]:


feature_cols = ['CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)',
                'PT08.S5(O3)','T','AH']
sns.pairplot(new_data[feature_cols], size=10)
plt.show()


# ### **Model, predict and solve** -- **Linear Regression**

# In[25]:


X = new_data.drop(columns=['RH'])
Y = new_data['RH']
X.shape, Y.shape


# In[26]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[27]:


linreg = LinearRegression()
linreg.fit(x_train,y_train)


# In[28]:


y_linpred = linreg.predict(x_test)


# In[29]:


linreg.score(x_train, y_train)


# In[30]:


print(linreg.intercept_)
list(zip(feature_cols,linreg.coef_))


# In[31]:


## RMSE(Root Mean Squared Error)
print('the RMSE value : {}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_linpred))))

## R Squared value or coefficient of determination
print('the R Squared value : {}'.format(metrics.r2_score(y_test,y_linpred)))

## Mean Absolute Error
print('the Mean Absolute Error : {}'.format(metrics.mean_absolute_error(y_test,y_linpred)))


# ### **Normalizing the input columns**

# In[32]:


#Standardizing the input
from sklearn.preprocessing import StandardScaler

## Please scale your input data before doing PCA in order to remove different variance across all the variables
ss = StandardScaler()
x = ss.fit_transform(X)


# In[33]:


## Quick check the standard deviation & mean
print(x.std())
print(x.mean())


# In[34]:


from sklearn.decomposition import PCA
plt.rcParams['figure.figsize'] = (8, 6)

pca=PCA().fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.annotate('11',xy=(11, .95))### We want total 95% variance and corresponding PC
plt.show()


# In[35]:


new_X = PCA(n_components=11).fit_transform(X)


# In[36]:


X_train,X_test,Y_train,Y_test = train_test_split(new_X,Y,test_size=0.20,random_state=123)
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[37]:


linReg = LinearRegression()
linReg.fit(X_train,Y_train)


# In[38]:


y_linPred = linReg.predict(X_test)
linReg.score(X_train,Y_train)


# In[39]:


print(linReg.intercept_)
print(linReg.coef_)


# In[40]:


## RMSE(Root Mean Squared Error)
print('the RMSE value : {}'.format(np.sqrt(metrics.mean_squared_error(Y_test,y_linPred))))

## R Squared value or coefficient of determination
print('the R Squared value : {}'.format(metrics.r2_score(Y_test,y_linPred)))

## Mean Absolute Error
print('the Mean Absolute Error : {}'.format(metrics.mean_absolute_error(Y_test,y_linPred)))


# ### **K Means**

# In[41]:


ssw = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x_train)
    ssw.append(sum(np.min(cdist(x_train, kmeanModel.cluster_centers_, 'cityblock'), axis=1)) / x_train.shape[0])


# In[42]:


plt.plot(K, ssw, 'bx-')
plt.xlabel('k')
plt.ylabel('ssw')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[43]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train)
kmeans.cluster_centers_


# In[44]:


size = kmeans.labels_
size = list(size)
size


# In[45]:


kmeans.predict(x_test)


# In[46]:


kmeans.score(x_train,y_train)


# ### Time Series

# In[47]:


#time_data = new_data.drop(columns='DT_num')
#time_data.set_index('Date_Time',inplace=True)


# In[65]:


plt.rcParams['figure.figsize'] = (20, 8)

# Let us see the original data spread
plt.plot(new_data.RH)


# In[66]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(new_data.RH, model = "additive")
decomposition.plot()
plt.show()


# In[67]:


log_air_RH = np.log(new_data.RH)

# Data after log normalizing - Constant variance
plt.plot(log_air_RH)


# In[68]:


log_air_RH_diff = log_air_RH - log_air_RH.shift()
log_air_RH_diff.dropna(inplace=True)
log_air_RH.dropna(inplace=True)


# In[69]:


# Data after log normalizing & 1st order differencing- Constant variance & mean - Removal of trend
plt.plot(log_air_RH_diff)


# In[70]:


from statsmodels.tsa.stattools import adfuller, acf, pacf
lag_acf = acf(log_air_RH_diff.values, nlags = 20)
lag_pacf = pacf(log_air_RH_diff.values, nlags = 20)

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(log_air_RH_diff)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(log_air_RH_diff)),linestyle='--')

# look at where the plot passes the upper confidence interval for the first time that gives the q value for ACF
#q = 4.75


# In[71]:


plt.subplot(121) 
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(log_air_RH_diff)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(log_air_RH_diff)),linestyle='--')

# look at where the plot passes the upper confidence interval for the first time for PACF and it gives us the p value
#p = 3


# In[72]:


from statsmodels.tsa.arima_model import ARIMA

# AR model
ar_model = ARIMA(log_air_RH, order=(2, 1, 0))  
results_AR = ar_model.fit(disp=-1)  
plt.plot(log_air_RH_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-log_air_RH_diff)**2))


# In[83]:


# MA model
ma_model = ARIMA(log_air_RH, order=(0, 1, 2))  
results_MA = ma_model.fit(disp=-1)  
plt.plot(log_air_RH_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-log_air_RH_diff)**2))


# In[74]:


# ARIMA model
arima_model = ARIMA(log_air_RH, order=(2, 1, 2))  
results_ARIMA = arima_model.fit(disp=-1)  
plt.plot(log_air_RH_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-log_air_RH_diff)**2))


# In[75]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


# In[76]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[77]:


predictions_ARIMA_log = pd.Series(log_air_RH.ix[0], index=log_air_RH.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[78]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(new_data)
plt.plot(predictions_ARIMA)


# In[79]:


predictions_ARIMA.head()


# In[80]:


new_data.head()


# In[81]:


forecast_error = new_data.RH - predictions_ARIMA
print('Mean Error : {}'.format(np.mean(forecast_error)))
print('Mean Squared Error : {}'.format(np.mean(forecast_error ** 2)))
print('Root Mean Squared Error : {}'.format(np.sqrt(np.mean(forecast_error ** 2))))
print('Mean Absolute Error : {}'.format(np.mean(abs(forecast_error))))


# ### Summary
# 
# #### Linear Regression :
# - RMSE value : 8.751927075999163
# - R Squared value : 0.967045075662062
# - Mean Absolute Error : 6.774707683807781
# 
# #### Time Series:
# - Mean Error : 12.595304082818268
# - Mean Squared Error : 1300.820099722775
# - Root Mean Squared Error : 36.06688369852287
# - Mean Absolute Error : 29.682955028632133
# 
# #### K-Means:
# Its not suitable model for prediction as accuracy score is -2264205580.848714 which is not desirable.
# 
# On basis of RMSE, mean absolute error n etc, we can say Linear Regression has less error values. So Linear Regression is suitable model.
