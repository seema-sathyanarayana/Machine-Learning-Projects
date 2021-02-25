#!/usr/bin/env python
# coding: utf-8

# ### Importing and Reading CSV

# #### Import Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error


# #### Reading the CSV file and changing Order Date to yyyy-mm format

# In[2]:


gs_df = pd.read_csv('Global2.csv')
gs_df['Order Date'] = pd.to_datetime(gs_df['Order Date'], format='%d-%m-%Y').dt.strftime('%Y-%m')
gs_df.head()


# ### Data preparation

# #### Changing the Order Date type to TimeStamp

# In[3]:


gs_df['Order Date'] = pd.to_datetime(gs_df['Order Date'], format='%Y-%m')
gs_df.info()


# #### Creating a new column Market Segment by concatenating columns Market and Segment

# In[4]:


gs_df['Market Segment'] = gs_df.Market + '-' + gs_df.Segment
gs_df.head()


# #### Grouping data by Order Date

# In[5]:


gs_tab = pd.pivot_table(gs_df, index=['Order Date'], values=['Profit'], columns=['Market Segment'], aggfunc=sum)
gs_tab


# #### Splitting data into train and test data sets

# In[6]:


train_len = 42
train_df = gs_tab[:train_len]
test_df = gs_tab[train_len:]


# #### Coefficient of Variance (CoV) calculation for every Market Segment

# In[7]:


train_df.std()/train_df.mean()


# **Observation**
# - Based on the Above table and graphs, the `most profitable` Market Segment is `APAC-Consumer`. 
# - `APAC-Consumer` has `lowest CoV` value when compared to other Market Segment.

# #### Picking the data specific to APAC-Consumer Market Segment and creating a ew dataframe

# In[8]:


APAC_Consumer = gs_df.loc[gs_df['Market Segment'] == 'APAC-Consumer']
APAC_Consumer = pd.pivot_table(APAC_Consumer, index=['Order Date'], values=['Sales', 'Quantity', 'Profit'], aggfunc=sum)
APAC_Consumer.head()


# ### Time Series Analysis for Sales forecast 

# #### Time Series plot for Sales

# In[9]:


APAC_Consumer.plot(y='Sales', figsize=(15,4))
plt.legend(loc='best')
plt.title('Sales data')
plt.show()


# ### Time Series Decomposition
# #### 1. Additive

# In[10]:


from pylab import rcParams
import statsmodels.api as sm

rcParams['figure.figsize'] = 12, 6
decomposition = sm.tsa.seasonal_decompose(APAC_Consumer.Sales, model='additive') # additive seasonal index
fig = decomposition.plot()
plt.show()


# #### 2. Multiplicative

# In[11]:


rcParams['figure.figsize'] = 12,6
decomposition = sm.tsa.seasonal_decompose(APAC_Consumer.Sales, model='multiplicative')
decomposition.plot()
plt.show()


# #### Splitting data into train and test data set

# In[12]:


train_len = 42
train_df = APAC_Consumer[:train_len]
test_df = APAC_Consumer[train_len:]


# ### Smoothing Techniques
# #### 1. Naive Method

# In[13]:


y_naive = test_df.copy()
y_naive['Naive Forecast'] = train_df['Sales'][train_len-1]


# In[14]:


## plot train, test and forecast
plt.figure(figsize=(16,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_naive['Naive Forecast'], label='Naive forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.show()


# In[15]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_naive['Naive Forecast'])/test_df['Sales'])*100, 2)

naive_result = pd.DataFrame({'Method': ['Naive method'], 'MAPE': mape})
naive_result


# #### 2. Simple Average Method

# In[16]:


y_avg = test_df.copy()
y_avg['Avg Forecast'] = train_df['Sales'].mean()


# In[17]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_avg['Avg Forecast'], label='Simple Average forecast')
plt.legend(loc='best')
plt.title('Simple Average Method')
plt.show()


# In[18]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_avg['Avg Forecast'])/test_df['Sales'])*100, 2)

avg_result = pd.DataFrame({'Method': ['Simple Average method'], 'MAPE': mape})
results = pd.concat([naive_result, avg_result])
results


# #### 3. Simple Moving Average Method

# In[19]:


y_sma = APAC_Consumer.copy()
ma_window = 3
y_sma['sma forecast'] = APAC_Consumer['Sales'].rolling(ma_window).mean()
y_sma['sma forecast'][train_len:] = y_sma['sma forecast'][train_len]


# In[20]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_sma['sma forecast'][train_len:], label='Simple Moving Average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()


# In[21]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_sma['sma forecast'][train_len:])/test_df['Sales'])*100, 2)

sma_result = pd.DataFrame({'Method': ['Simple Moving Average method'], 'MAPE': mape})
results = pd.concat([results, sma_result])
results


# #### 4. Simple Exponential Smoothing Technique

# In[22]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(train_df['Sales'])
model_fit = model.fit(smoothing_level=0.5)
model_fit.params


# In[23]:


y_ses = test_df.copy()
y_ses['SES Forecast'] = model_fit.forecast(6) 


# In[24]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_ses['SES Forecast'], label='Simple exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Simple exponential smoothing technique')
plt.show()


# In[25]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_ses['SES Forecast'])/test_df['Sales'])*100, 2)

ses_result = pd.DataFrame({'Method': ['Simple exponential smoothing'], 'MAPE': mape})
results = pd.concat([results, ses_result])
results


# #### 5. Holt's exponential smoothing technique

# In[26]:


from statsmodels.tsa.holtwinters import  ExponentialSmoothing

model = ExponentialSmoothing(np.array(train_df['Sales']), seasonal_periods=5, trend='additive', seasonal=None)
model_fit = model.fit(smoothing_level=0.1, smoothing_slope=0.1, optimized=False)
model_fit.params


# In[27]:


y_holt = test_df.copy()
y_holt['hes forecast'] = model_fit.forecast(6)


# In[28]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_holt['hes forecast'], label='Holts exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Holts exponential smoothing technique')
plt.show()


# In[29]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_holt['hes forecast'])/test_df['Sales'])*100, 2)

holt_result = pd.DataFrame({'Method': ['Holt exponential smoothing'], 'MAPE': mape})
results = pd.concat([results, holt_result])
results


# #### 6. Holt-Winter's Additive Method (Trend+Seassonality)

# In[30]:


y_hwa = test_df.copy()
model = ExponentialSmoothing(np.array(test_df['Sales']), seasonal_periods=5, trend='add', seasonal='add')
model_fit = model.fit(smoothing_level=0.001, smoothing_slope=0.002, smoothing_seasonal=0.001, optimized=False)
model_fit.params


# In[31]:


y_hwa['HWA Forecast'] = model_fit.forecast(6)


# In[32]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_hwa['HWA Forecast'], label='Holts-Winter additive smoothing forecast')
plt.legend(loc='best')
plt.title('Holts-Winter additive smoothing technique')
plt.show()


# In[33]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_hwa['HWA Forecast'])/test_df['Sales'])*100, 2)

hwa_result = pd.DataFrame({'Method': ['Holts-Winter Additive smoothing'], 'MAPE': mape})
results = pd.concat([results, hwa_result])
results


# #### 7. Holt-Winter's Multiplicative Method (Trend+Seassonality)

# In[34]:


y_hwm = test_df.copy()
model = ExponentialSmoothing(np.array(test_df['Sales']), seasonal_periods=5, trend='add', seasonal='mul')
model_fit = model.fit(smoothing_level=0.001, smoothing_slope=0.002, smoothing_seasonal=0.001, optimized=False)
model_fit.params


# In[35]:


y_hwm['HWM Forecast'] = model_fit.forecast(6)


# In[36]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_hwm['HWM Forecast'], label='Holts-Winter multiplicative smoothing forecast')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.title('Holts-Winter multiplicative smoothing technique')
plt.show()


# In[37]:


mape = np.round(np.mean(np.abs(test_df['Sales'] - y_hwm['HWM Forecast'])/test_df['Sales'])*100, 2)

hwm_result = pd.DataFrame({'Method': ['Holts-Winter Multiplicative smoothing'], 'MAPE': mape})
results = pd.concat([results, hwm_result])
results


# ### Stationary vs Non Stationary

# In[38]:


APAC_Consumer.plot(y='Sales', figsize=(15,4))
plt.legend(loc='best')
plt.title('Sales data')
plt.show()


# #### Augmented Dickey-Fuller (ADF) test

# In[39]:


from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(APAC_Consumer['Sales'])

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values at 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# #### KPSS test

# In[40]:


from statsmodels.tsa.stattools import kpss

kpss_test = kpss(APAC_Consumer['Sales'])

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values at 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# #### Box Cox transformation to make variance constant

# In[41]:


from scipy.stats import boxcox

data_boxcox = pd.Series(boxcox(APAC_Consumer['Sales'], lmbda=0), index = APAC_Consumer.index)

plt.figure(figsize=(15,4))
plt.plot(data_boxcox, label='After Box Cox tranformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()


# #### Differencing to remove trend

# In[42]:


data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), APAC_Consumer.index)
plt.figure(figsize=(15,4))
plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
plt.legend(loc='best')
plt.title('After Box Cox transform and differencing')
plt.show()


# In[43]:


data_boxcox_diff.dropna(inplace=True)


# In[44]:


data_boxcox_diff.tail()


# #### Augmented Dickey-Fuller (ADF) test

# In[45]:


from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(data_boxcox_diff)

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values at 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# #### KPSS test

# In[46]:


from statsmodels.tsa.stattools import kpss

kpss_test = kpss(data_boxcox_diff)

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values at 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# ### Autocorrelation
# #### Autocorrelation function (ACF)

# In[47]:


from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(15,4))
plot_acf(data_boxcox_diff, ax=plt.gca(), lags=30)
plt.show()


# #### Partial Autocorrelation Function (PACF)

# In[48]:


from statsmodels.graphics.tsaplots import plot_pacf

plt.figure(figsize=(15,4))
plot_pacf(data_boxcox_diff, ax=plt.gca(), lags=30)
plt.show()


# #### Split into train and test data set

# In[49]:


train_data_boxcox = data_boxcox[:train_len]
test_data_boxcox = data_boxcox[train_len:]
train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
test_data_boxcox_diff = data_boxcox_diff[train_len-1:]


# ### ARIMA set of techniques
# #### 1. Auto Regression model (AR)

# In[50]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0)) 
model_fit = model.fit()
print(model_fit.params)


# In[51]:


# Recover original time series
y_hat_ar = data_boxcox_diff.copy()
y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum()
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(data_boxcox[0])
y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox'])


# In[52]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_hat_ar['ar_forecast'][test_df.index.min():], label='Auto regression forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
plt.show()


# In[53]:


mape = np.round(np.mean(np.abs(test_df['Sales']-y_hat_ar['ar_forecast'][test_df.index.min():])/test_df['Sales'])*100,2)

ar_results = pd.DataFrame({'Method': ['Autoregressive (AR) method'], 'MAPE': mape })
results = pd.concat([results, ar_results])
results


# #### 2. Moving average method (MA)

# In[54]:


model = ARIMA(train_data_boxcox_diff, order=(0, 0, 1)) 
model_fit = model.fit()
print(model_fit.params)


# In[55]:


# Recover original time series
y_hat_ma = data_boxcox_diff.copy()
y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'].cumsum()
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].add(data_boxcox[0])
y_hat_ma['ma_forecast'] = np.exp(y_hat_ma['ma_forecast_boxcox'])


# In[56]:


# Plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_hat_ma['ma_forecast'][test_df.index.min():], label='Moving average forecast')
plt.legend(loc='best')
plt.title('Moving average Method')
plt.show()


# In[57]:


mape = np.round(np.mean(np.abs(test_df['Sales']-y_hat_ma['ma_forecast'][test_df.index.min():])/test_df['Sales'])*100,2)

ma_results = pd.DataFrame({'Method': ['Moving Average (MA) method'], 'MAPE': mape})
results = pd.concat([results, ma_results])
results


# #### 3. Auto regression moving average method (ARMA)

# In[58]:


model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.params)


# In[59]:


# Recover original time series
y_hat_arma = data_boxcox_diff.copy()
y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'].cumsum()
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].add(data_boxcox[0])
y_hat_arma['arma_forecast'] = np.exp(y_hat_arma['arma_forecast_boxcox'])


# In[60]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot( APAC_Consumer['Sales'][:train_len-1], label='Train')
plt.plot(APAC_Consumer['Sales'][train_len-1:], label='Test')
plt.plot(y_hat_arma['arma_forecast'][test_df.index.min():], label='ARMA forecast')
plt.legend(loc='best')
plt.title('ARMA Method')
plt.show()


# In[61]:


mape = np.round(np.mean(np.abs(test_df['Sales']-y_hat_arma['arma_forecast'][train_len-1:])/test_df['Sales'])*100,2)

arma_results = pd.DataFrame({'Method': ['Autoregressive moving average (ARMA) method'], 'MAPE': mape})
results = pd.concat([results, arma_results])
results


# #### 4. Auto regressive integrated moving average (ARIMA)

# In[62]:


model = ARIMA(train_data_boxcox, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.params)


# In[63]:


# Recover original time series forecast
y_hat_arima = data_boxcox_diff.copy()
y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(data_boxcox[0])
y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])


# In[64]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_hat_arima['arima_forecast'][test_df.index.min():], label='ARIMA forecast')
plt.legend(loc='best')
plt.title('Autoregressive integrated moving average (ARIMA) method')
plt.show()


# In[65]:


mape = np.round(np.mean(np.abs(test_df['Sales']-y_hat_arima['arima_forecast'][test_df.index.min():])/test_df['Sales'])*100,2)

arima_results = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'MAPE': mape})
results = pd.concat([results, arima_results])
results


# #### 5. Seasonal auto regressive integrated moving average (SARIMA)

# In[66]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 6)) 
model_fit = model.fit()
print(model_fit.params)


# In[67]:


# Recover original time series forecast
y_hat_sarima = data_boxcox_diff.copy()
y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])


# In[68]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train_df['Sales'], label='Train')
plt.plot(test_df['Sales'], label='Test')
plt.plot(y_hat_sarima['sarima_forecast'][test_df.index.min():], label='SARIMA forecast')
plt.legend(loc='best')
plt.title('Seasonal autoregressive integrated moving average (SARIMA) method')
plt.show()


# In[69]:


mape = np.round(np.mean(np.abs(test_df['Sales']-y_hat_sarima['sarima_forecast'][test_df.index.min():])/test_df['Sales'])*100,2)

sarima_results = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'MAPE': mape})
results = pd.concat([results, sarima_results])
sale_results = results
sale_results


# **Observation**
# - The technique which works best for sales forecast is SARIMA as it has less MAPE value compared to others.
# - From graph also we can see that forecast is most accurate with SARIMA.

# ### Time Series Analysis for Quantity forecast
# #### Time Series plot for Quantity

# In[70]:


APAC_Consumer.plot(y='Quantity', figsize=(15,4))
plt.legend(loc='best')
plt.title('Quantity data')
plt.show()


# ### Time Series Decomposition
# #### 1. Additive

# In[71]:


rcParams['figure.figsize'] = 12,6
decomposition = sm.tsa.seasonal_decompose(APAC_Consumer['Quantity'], model='additive')
decomposition.plot()
plt.show()


# #### 2. Multiplicative

# In[72]:


rcParams['figure.figsize'] = 12,6
decomposition = sm.tsa.seasonal_decompose(APAC_Consumer['Quantity'], model='multiplicative')
decomposition.plot()
plt.show()


# #### Train and test split

# In[73]:


train_len = 42
train_df = APAC_Consumer[0:train_len]
test_df = APAC_Consumer[train_len:]


# ### Smoothing Techniques
# #### 1. Naive Method

# In[74]:


y_naive = test_df.copy()
y_naive['Naive Forecast'] = train_df['Quantity'][train_len-1]


# In[75]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_naive['Naive Forecast'], label='Naive forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.show()


# In[76]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_naive['Naive Forecast'])/test_df['Quantity'])*100, 2)

naive_result = pd.DataFrame({'Method': ['Naive method'], 'MAPE': mape})
naive_result


# #### 2. Simple Average Method

# In[77]:


y_avg = test_df.copy()
y_avg['Avg Forecast'] = train_df['Quantity'].mean()


# In[78]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_avg['Avg Forecast'], label='Simple Average forecast')
plt.legend(loc='best')
plt.title('Simple Average Method')
plt.show()


# In[79]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_avg['Avg Forecast'])/test_df['Quantity'])*100, 2)

avg_result = pd.DataFrame({'Method': ['Simple Average method'], 'MAPE': mape})
results = pd.concat([naive_result, avg_result])
results


# #### 3. Simple Moving Average Method

# In[80]:


y_sma = APAC_Consumer.copy()
ma_window = 3
y_sma['sma forecast'] = APAC_Consumer['Quantity'].rolling(ma_window).mean()
y_sma['sma forecast'][train_len:] = y_sma['sma forecast'][train_len]


# In[81]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_sma['sma forecast'][train_len:], label='Simple Moving Average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()


# In[82]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_sma['sma forecast'][train_len:])/test_df['Quantity'])*100, 2)

sma_result = pd.DataFrame({'Method': ['Simple Moving Average method'], 'MAPE': mape})
results = pd.concat([results, sma_result])
results


# #### 4. Simple Exponential Smoothing Technique

# In[83]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(train_df['Quantity'])
model_fit = model.fit(smoothing_level=0.5)
model_fit.params


# In[84]:


y_ses = test_df.copy()
y_ses['SES Forecast'] = model_fit.forecast(6) # no months we need to forecast


# In[85]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_ses['SES Forecast'], label='Simple exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Simple exponential smoothing technique')
plt.show()


# In[86]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_ses['SES Forecast'])/test_df['Quantity'])*100, 2)

ses_result = pd.DataFrame({'Method': ['Simple exponential smoothing'], 'MAPE': mape})
results = pd.concat([results, ses_result])
results


# #### 5. Holt's exponential smoothing technique

# In[87]:


from statsmodels.tsa.holtwinters import  ExponentialSmoothing

model = ExponentialSmoothing(np.array(train_df['Quantity']), seasonal_periods=5, trend='additive', seasonal=None)
model_fit = model.fit()
model_fit.params


# In[88]:


y_holt = test_df.copy()
y_holt['hes forecast'] = model_fit.forecast(6)


# In[89]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_holt['hes forecast'], label='Holts exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Holts exponential smoothing technique')
plt.show()


# In[90]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_holt['hes forecast'])/test_df['Quantity'])*100, 2)

holt_result = pd.DataFrame({'Method': ['Holt exponential smoothing'], 'MAPE': mape})
results = pd.concat([results, holt_result])
results


# #### 6. Holt-Winter's Additive Method (Trend+Seassonality)

# In[91]:


y_hwa = test_df.copy()
model = ExponentialSmoothing(np.array(test_df['Quantity']), seasonal_periods=5, trend='add', seasonal='add')
model_fit = model.fit()
model_fit.params


# In[92]:


y_hwa['HWA Forecast'] = model_fit.forecast(6)


# In[93]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_hwa['HWA Forecast'], label='Holts-Winter additive smoothing forecast')
plt.legend(loc='best')
plt.title('Holts-Winter additive smoothing technique')
plt.show()


# In[94]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_hwa['HWA Forecast'])/test_df['Quantity'])*100, 2)

hwa_result = pd.DataFrame({'Method': ['Holts-Winter Additive smoothing'], 'MAPE': mape})
results = pd.concat([results, hwa_result])
results


# #### 7. Holt-Winter's Multiplicative Method (Trend+Seassonality)

# In[95]:


y_hwm = test_df.copy()
model = ExponentialSmoothing(np.array(test_df['Quantity']), seasonal_periods=5, trend='add', seasonal='mul')
model_fit = model.fit()
model_fit.params


# In[96]:


y_hwm['HWM Forecast'] = model_fit.forecast(6)


# In[97]:


## plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_hwm['HWM Forecast'], label='Holts-Winter multiplicative smoothing forecast')
plt.legend(loc='best')
plt.title('Holts-Winter multiplicative smoothing technique')
plt.show()


# In[98]:


mape = np.round(np.mean(np.abs(test_df['Quantity'] - y_hwm['HWM Forecast'])/test_df['Quantity'])*100, 2)

hwm_result = pd.DataFrame({'Method': ['Holts-Winter Multiplicative smoothing'], 'MAPE': mape})
results = pd.concat([results, hwm_result])
results


# ### Stationary vs Non Stationary

# In[99]:


APAC_Consumer.plot(y='Quantity', figsize=(15,4))
plt.legend(loc='best')
plt.title('Quantity data')
plt.show()


# #### Augmented Dickey-Fuller (ADF) test

# In[100]:


from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(APAC_Consumer['Quantity'])

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values at 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# #### KPSS test

# In[101]:


from statsmodels.tsa.stattools import kpss

kpss_test = kpss(APAC_Consumer['Quantity'])

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values at 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# #### Box Cox transformation to make variance constant

# In[102]:


from scipy.stats import boxcox

data_boxcox = pd.Series(boxcox(APAC_Consumer['Quantity'], lmbda=0), index = APAC_Consumer.index)

plt.figure(figsize=(15,4))
plt.plot(data_boxcox, label='After Box Cox tranformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()


# #### Differencing to remove trend

# In[103]:


data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), APAC_Consumer.index)
plt.figure(figsize=(15,4))
plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
plt.legend(loc='best')
plt.title('After Box Cox transform and differencing')
plt.show()


# In[104]:


data_boxcox_diff.dropna(inplace=True)


# In[105]:


data_boxcox_diff.tail()


# #### Augmented Dickey-Fuller (ADF) test

# In[106]:


from statsmodels.tsa.stattools import adfuller

adf_test = adfuller(data_boxcox_diff)

print('ADF Statistic: %f' % adf_test[0])
print('Critical Values at 0.05: %.2f' % adf_test[4]['5%'])
print('p-value: %f' % adf_test[1])


# #### KPSS test

# In[107]:


from statsmodels.tsa.stattools import kpss

kpss_test = kpss(data_boxcox_diff)

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values at 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])


# ### Autocorrelation
# #### Autocorrelation function (ACF)

# In[108]:


from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(15,4))
plot_acf(data_boxcox_diff, ax=plt.gca(), lags=30)
plt.show()


# #### Partial Autocorrelation Function (PACF)

# In[109]:


from statsmodels.graphics.tsaplots import plot_pacf

plt.figure(figsize=(15,4))
plot_pacf(data_boxcox_diff, ax=plt.gca(), lags=30)
plt.show()


# In[110]:


train_data_boxcox = data_boxcox[:train_len]
test_data_boxcox = data_boxcox[train_len:]
train_data_boxcox_diff = data_boxcox_diff[:train_len-1]
test_data_boxcox_diff = data_boxcox_diff[train_len-1:]


# ### ARIMA set of techniques
# #### 1. Auto Regression model (AR)

# In[111]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_data_boxcox_diff, order=(1, 0, 0)) 
model_fit = model.fit()
print(model_fit.params)


# In[112]:


# Recover original time series
y_hat_ar = data_boxcox_diff.copy()
y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'].cumsum()
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].add(data_boxcox[0])
y_hat_ar['ar_forecast'] = np.exp(y_hat_ar['ar_forecast_boxcox'])


# In[113]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_hat_ar['ar_forecast'][test_df.index.min():], label='Auto regression forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
plt.show()


# In[114]:


mape = np.round(np.mean(np.abs(test_df['Quantity']-y_hat_ar['ar_forecast'][test_df.index.min():])/test_df['Quantity'])*100,2)

ar_results = pd.DataFrame({'Method': ['Autoregressive (AR) method'], 'MAPE': mape })
results = pd.concat([results, ar_results])
results


# #### 2. Moving average method (MA)

# In[115]:


model = ARIMA(train_data_boxcox_diff, order=(0, 0, 1)) 
model_fit = model.fit()
print(model_fit.params)


# In[116]:


# Recover original time series
y_hat_ma = data_boxcox_diff.copy()
y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'].cumsum()
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].add(data_boxcox[0])
y_hat_ma['ma_forecast'] = np.exp(y_hat_ma['ma_forecast_boxcox'])


# In[117]:


# Plot train, test and forecast
plt.figure(figsize=(15,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_hat_ma['ma_forecast'][test_df.index.min():], label='Moving average forecast')
plt.legend(loc='best')
plt.title('Moving average Method')
plt.show()


# In[118]:


mape = np.round(np.mean(np.abs(test_df['Quantity']-y_hat_ma['ma_forecast'][test_df.index.min():])/test_df['Quantity'])*100,2)

ma_results = pd.DataFrame({'Method': ['Moving Average (MA) method'], 'MAPE': mape})
results = pd.concat([results, ma_results])
results


# #### 3. Auto regression moving average method (ARMA)

# In[119]:


model = ARIMA(train_data_boxcox_diff, order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.params)


# In[120]:


# Recover original time series
y_hat_arma = data_boxcox_diff.copy()
y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'].cumsum()
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].add(data_boxcox[0])
y_hat_arma['arma_forecast'] = np.exp(y_hat_arma['arma_forecast_boxcox'])


# In[121]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot( APAC_Consumer['Quantity'][:train_len-1], label='Train')
plt.plot(APAC_Consumer['Quantity'][train_len-1:], label='Test')
plt.plot(y_hat_arma['arma_forecast'][test_df.index.min():], label='ARMA forecast')
plt.legend(loc='best')
plt.title('ARMA Method')
plt.show()


# In[122]:


mape = np.round(np.mean(np.abs(test_df['Quantity']-y_hat_arma['arma_forecast'][train_len-1:])/test_df['Quantity'])*100,2)

arma_results = pd.DataFrame({'Method': ['Autoregressive moving average (ARMA) method'], 'MAPE': mape})
results = pd.concat([results, arma_results])
results


# #### 4. Auto regressive integrated moving average (ARIMA)

# In[123]:


model = ARIMA(train_data_boxcox, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.params)


# In[124]:


# Recover original time series forecast
y_hat_arima = data_boxcox_diff.copy()
y_hat_arima['arima_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox_diff'].cumsum()
y_hat_arima['arima_forecast_boxcox'] = y_hat_arima['arima_forecast_boxcox'].add(data_boxcox[0])
y_hat_arima['arima_forecast'] = np.exp(y_hat_arima['arima_forecast_boxcox'])


# In[125]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_hat_arima['arima_forecast'][test_df.index.min():], label='ARIMA forecast')
plt.legend(loc='best')
plt.title('Autoregressive integrated moving average (ARIMA) method')
plt.show()


# In[126]:


mape = np.round(np.mean(np.abs(test_df['Quantity']-y_hat_arima['arima_forecast'][test_df.index.min():])/test_df['Quantity'])*100,2)

arima_results = pd.DataFrame({'Method':['Autoregressive integrated moving average (ARIMA) method'], 'MAPE': mape})
results = pd.concat([results, arima_results])
results


# #### 5. Seasonal auto regressive integrated moving average (SARIMA)

# In[127]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data_boxcox, order=(1, 1, 1), seasonal_order=(1, 1, 1, 6)) 
model_fit = model.fit()
print(model_fit.params)


# In[128]:


# Recover original time series forecast
y_hat_sarima = data_boxcox_diff.copy()
y_hat_sarima['sarima_forecast_boxcox'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_sarima['sarima_forecast'] = np.exp(y_hat_sarima['sarima_forecast_boxcox'])


# In[129]:


# Plot train, test and forecast
plt.figure(figsize=(12,4))
plt.plot(train_df['Quantity'], label='Train')
plt.plot(test_df['Quantity'], label='Test')
plt.plot(y_hat_sarima['sarima_forecast'][test_df.index.min():], label='SARIMA forecast')
plt.legend(loc='best')
plt.title('Seasonal autoregressive integrated moving average (SARIMA) method')
plt.show()


# In[130]:


mape = np.round(np.mean(np.abs(test_df['Quantity']-y_hat_sarima['sarima_forecast'][test_df.index.min():])/test_df['Quantity'])*100,2)

sarima_results = pd.DataFrame({'Method':['Seasonal autoregressive integrated moving average (SARIMA) method'], 'MAPE': mape})
results = pd.concat([results, sarima_results])
quantity_results = results
quantity_results


# **Observation**
# - The technique which works best for quantity forecast is SARIMA as it has less MAPE value compared to others.
# - From graph also we can see that forecast is most accurate with SARIMA.
