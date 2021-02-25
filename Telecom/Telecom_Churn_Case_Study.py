#!/usr/bin/env python
# coding: utf-8

# 
# # Telecom Churn - Case Study
# 
# 
# ## Business Problem Overview
# 
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# For many incumbent operators, retaining high profitable customers is the number one business goal.
# 
# 
# To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# 
# ## Business Objective
# 
#  The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. 
# 
# 
# The usiness objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behaviour during churn will be helpful.
# 

# ### Importing the Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ### Reading the CSV file

# In[2]:


df = pd.read_csv('telecom_churn_data.csv')
df.head()


# ### Understanding the data

# In[3]:


# Information on dataframe

df.info()


# In[4]:


# Description of numerical variables of dataframe

df.describe()


# In[5]:


# Description of categorical variables of dataframe

df.describe(include=object)


# 
# ## Data Preparation

# In[6]:


# Columns having more than and equal to 70 percent of Null values 

df.columns[round(100*df.isnull().sum()/len(df),2)>=70]


# In[7]:


# Dropping columns having more than and equal 70 percentage null values

df.drop(['fb_user_6', 'fb_user_7', 'fb_user_8', 'fb_user_9', 'date_of_last_rech_data_6', 'date_of_last_rech_data_7',
         'date_of_last_rech_data_8', 'date_of_last_rech_data_9', 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8',
         'max_rech_data_9', 'count_rech_2g_6', 'count_rech_2g_7', 'count_rech_2g_8', 'count_rech_2g_9', 'count_rech_3g_6',
         'count_rech_3g_7', 'count_rech_3g_8', 'count_rech_3g_9', 'arpu_3g_6', 'arpu_3g_7', 'arpu_3g_8', 'arpu_3g_9',
         'arpu_2g_6', 'arpu_2g_7', 'arpu_2g_8', 'arpu_2g_9', 'night_pck_user_6', 'night_pck_user_7', 'night_pck_user_8',
         'night_pck_user_9'], axis=1, inplace=True)
df.info()


# In[8]:


# Filling missing values with 0

df['total_rech_data_6'].fillna(0, inplace=True)
df['total_rech_data_7'].fillna(0, inplace=True)
df['total_rech_data_8'].fillna(0, inplace=True)
df['total_rech_data_9'].fillna(0, inplace=True)
df['av_rech_amt_data_6'].fillna(0, inplace=True)
df['av_rech_amt_data_7'].fillna(0, inplace=True)
df['av_rech_amt_data_8'].fillna(0, inplace=True)
df['av_rech_amt_data_9'].fillna(0, inplace=True)


# In[9]:


# Checking for columns with less than 70 percent null values

df_lt_70 = df.columns[round(100*df.isnull().sum()/len(df),2)<70]
df_lt_70


# In[10]:


# Description of numerical variables of dataframe with columns less null values

df[df_lt_70].describe()


# In[11]:


# Description of categorical variables of dataframe with column less null values

df[df_lt_70].describe(include='object')


# ### Imputation

# In[12]:


# Dataframe containing filtered data after removal of columns having >70 % of null values

df=df[df_lt_70]


# In[13]:


df.shape


# In[14]:


#columns which are numeric 

colsnumeric = df.select_dtypes([np.int64,np.float64]).columns


# In[15]:


df[colsnumeric].head(10)


# In[16]:


#columns which are objects which are categorical

colsobject = df.select_dtypes([np.object]).columns


# In[17]:


df[colsobject].head()


# In[18]:


### Use Simple Imputer with max_frequency for imputation for categorical variables

from sklearn.impute import SimpleImputer

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df[colsobject] = imp_mode.fit_transform(df[colsobject])


# In[19]:


### Use Simple Imputer with median for imputation for continuous variables

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

df[colsnumeric] = imp_median.fit_transform(df[colsnumeric])


# In[20]:


df.shape


# ### Derived Columns
# 
# - tot_data_rech = tot_rech * av_data_rech
# - tot_rech = tot_data_rech + tot_rech_amt

# In[21]:


# Creating columns to know the total data recharge in month of 6 and 7

df['tot_data_rech_6'] = df['total_rech_data_6'] * df['av_rech_amt_data_6']
df['tot_data_rech_7'] = df['total_rech_data_7'] * df['av_rech_amt_data_7']
df.head()


# In[22]:


# Creating column for average recharge for month 6 and 7 month

df['tot_rech_6'] = df['tot_data_rech_6'] + df['total_rech_amt_6']
df['tot_rech_7'] = df['tot_data_rech_7'] + df['total_rech_amt_7']
df['avg_rech_6-7'] = round((df['tot_rech_6'] + df['tot_rech_7'])/2, 2)
df.head()


# ### Deriving high value customers 

# In[23]:


# Obtaining the customers who have average recharge more than equal to 70th percentile

high_value_cust = df[df['avg_rech_6-7'] >= np.quantile(df['avg_rech_6-7'], 0.7)]
high_value_cust.head()


# In[24]:


high_value_cust.shape


# - We have obtained almost 30k data points for high valued customers
# - Total usage of customers are calculated in Sept month below to tag the churn

# In[25]:


high_value_cust['total_usage_9'] = high_value_cust['total_ic_mou_9'] + high_value_cust['total_og_mou_9'] +    high_value_cust['vol_2g_mb_9'] + high_value_cust['vol_3g_mb_9']


# In[26]:


high_value_cust.head()


# ### Tagging churn 

# In[27]:


def is_churn(x):
    if x == 0:
        return 1
    else:
        return 0

high_value_cust['churn'] = high_value_cust['total_usage_9'].apply(is_churn)


# In[28]:


churn_percentage = round(100 * high_value_cust['churn'].value_counts()/len(high_value_cust['churn']), 2)
churn_percentage


# - Around 8% of the customers in data have churned.
# - Dropping the columns belonging to Sept month below.

# In[29]:


churn_phase_col = ['last_date_of_month_9', 'arpu_9', 'onnet_mou_9', 'offnet_mou_9', 'roam_ic_mou_9', 'roam_og_mou_9',
                   'loc_og_t2t_mou_9', 'loc_og_t2m_mou_9', 'loc_og_t2f_mou_9', 'loc_og_t2c_mou_9', 'loc_og_mou_9',
                   'std_og_t2t_mou_9', 'std_og_t2m_mou_9', 'std_og_t2f_mou_9', 'std_og_t2c_mou_9', 'std_og_mou_9',
                   'isd_og_mou_9', 'spl_og_mou_9', 'og_others_9', 'total_og_mou_9', 'loc_ic_t2t_mou_9', 'loc_ic_t2m_mou_9',
                   'loc_ic_t2f_mou_9', 'loc_ic_mou_9', 'std_ic_t2t_mou_9', 'std_ic_t2m_mou_9', 'std_ic_t2o_mou_9',
                   'std_ic_t2f_mou_9', 'std_ic_mou_9', 'total_ic_mou_9', 'spl_ic_mou_9', 'isd_ic_mou_9', 'ic_others_9',
                   'total_rech_num_9', 'total_rech_amt_9', 'max_rech_amt_9', 'date_of_last_rech_9', 'last_day_rch_amt_9',
                   'total_rech_data_9', 'av_rech_amt_data_9', 'vol_2g_mb_9', 'vol_3g_mb_9', 'monthly_2g_9', 'sachet_2g_9',
                   'monthly_3g_9', 'sachet_3g_9', 'sep_vbc_3g', 'total_usage_9']

high_value_cust = high_value_cust.drop(churn_phase_col, axis=1)


# In[30]:


high_value_cust.head()


# In[31]:


high_value_cust.info()


# In[32]:


# Changing the data type to Object from int

high_value_cust.circle_id = high_value_cust.circle_id.astype(str)
high_value_cust.mobile_number = high_value_cust.mobile_number.astype(str)


# ### Outlier Treatment

# In[33]:


# checking the percentile values for numerical values to handle outliers

num_col = high_value_cust.select_dtypes([np.int64,np.float64]).columns

for i in num_col:
    print(i)
    val = np.quantile(high_value_cust[i], [0.5,0.7,0.8,0.9,0.95,1])
    print(val)


# In[34]:


# Outliers are clipped to 95th percentile

for i in num_col:
    high_value_cust[i] = np.clip(high_value_cust[i], high_value_cust[i].quantile([0.0, 0.95][0]),
                          high_value_cust[i].quantile([0.0, 0.95][1]))


# In[35]:


high_value_cust.describe()


# ## Data Modelling 
# 
# 
# ### 1) Logistic Regression

# ### Train test split of data

# In[36]:


from sklearn.model_selection import train_test_split

num_col = num_col.drop('churn', 1)
X = high_value_cust[num_col]
y = high_value_cust['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Scaling

# In[37]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[num_col] = scaler.fit_transform(X_train[num_col])


# ### Logistic Regression

# In[38]:


# Import 'LogisticRegression' and create a LogisticRegression object

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')


# In[39]:


# Import RFE and select 15 variables

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[40]:


# Let's take a look at which features have been selected by RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[41]:


# Put all the columns selected by RFE in the variable 'col'

col = X_train.columns[rfe.support_]
col


# In[42]:


# Import statsmodels

import statsmodels.api as sm

# Select only the columns selected by RFE

X_train = X_train[col]


# In[43]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[44]:


# Import 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[45]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[46]:


X_train.drop('arpu_7', axis=1, inplace=True)


# In[47]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[48]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[49]:


X_train.drop('loc_og_mou_8', axis=1, inplace=True)


# In[50]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[51]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[52]:


X_train.drop('total_og_mou_8', axis=1, inplace=True)


# In[53]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[54]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[55]:


X_train.drop('loc_ic_mou_8', axis=1, inplace=True)


# In[56]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[57]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[58]:


X_train.drop('loc_ic_t2m_mou_8', axis=1, inplace=True)


# In[59]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[60]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[61]:


X_train.drop('arpu_6', axis=1, inplace=True)


# In[62]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[63]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


### these are the top features which is impacting churn in logistic regression


# ### At this stage Pvalue of all the features are 0 and VIF is under 4 

# In[64]:


# Use 'predict' to predict the probabilities on the train set

y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[65]:


# Reshaping it into an array

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[66]:


# Create a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'churn':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[67]:


y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[68]:


# Import metrics from sklearn for evaluation

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score


# In[69]:


# Create confusion matrix 

confusion = confusion_matrix(y_train_pred_final.churn, y_train_pred_final.Predicted )
print(confusion)


# In[70]:


# Predicted     not_churn    churn
# Actual
# not_churn        19257      52
# churn            1634       57  


# In[71]:


# Let's check the overall accuracy

print(accuracy_score(y_train_pred_final.churn, y_train_pred_final.Predicted))


# In[72]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[73]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# - Finding optimal cutoff

# In[74]:


# ROC function

def draw_roc(actual, probs):
    fpr, tpr, thresholds = roc_curve(actual, probs,
                                              drop_intermediate = False)
    auc_score = roc_auc_score(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[75]:


fpr, tpr, thresholds = roc_curve( y_train_pred_final.churn, y_train_pred_final.Conversion_Prob, 
                                         drop_intermediate = False )


# In[76]:


# Call the ROC function

draw_roc(y_train_pred_final.churn, y_train_pred_final.Conversion_Prob)


# The area under the curve of the ROC is 0.88 which is quite good. So we seem to have a good model. Let's also check the sensitivity and specificity tradeoff to find the optimal cutoff point.

# In[77]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[78]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci','preci','recal'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_train_pred_final.churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    preci = cm1[1,1]/(cm1[1,1]+cm1[0,1])
    recal = cm1[1,1]/(cm1[1,1]+cm1[1,0])
    cutoff_df.loc[i] =[ i , accuracy, sensi, speci, preci, recal]
print(cutoff_df)


# In[79]:


from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_pred_final.churn, y_train_pred_final.Conversion_Prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[80]:


# Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[81]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.1 else 0)

y_train_pred_final.head()


# In[82]:


# Let's check the accuracy now


accuracyscore=accuracy_score(y_train_pred_final.churn, y_train_pred_final.final_predicted)
print(accuracyscore)


# In[83]:


# Let's create the confusion matrix once again

confusion2 = confusion_matrix(y_train_pred_final.churn, y_train_pred_final.final_predicted )
confusion2


# In[84]:


# Predicted     not_churn    churn
# Actual
# not_churn        15824     3485
# churn            346       1345  


# Here, ratio of predicted customers as churned to actually churned is greater when compared to the predicted customers as not churn to actually churned

# In[85]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[86]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[87]:


import pandas as pd 


results = pd.DataFrame({'Method':['Logistic Regression Train'], 'Accuracyscore': [accuracyscore], 'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})


results = results[['Method','Accuracyscore', 'Sensitivity','Specificity','precision', 'recall']]
results


# ## Prediction on Test dataset

# In[88]:


X_test[num_col] = scaler.transform(X_test[num_col])

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train.columns]


# In[89]:


# Add a constant to X_test

X_test_sm = sm.add_constant(X_test_new)


# In[90]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(X_test_sm)


# In[91]:


y_test_pred[:10]


# In[92]:


# Converting y_pred_1 to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[93]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[94]:


# Make predictions on the test set using 0.45 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.1 else 0)
y_pred_final.head()


# In[95]:


# Let's check the accuracy now



accuracyscore=accuracy_score(y_pred_final['churn'], y_pred_final.final_predicted)
print(accuracyscore)


# In[96]:


# Let's create the confusion matrix once again

confusion3 = confusion_matrix(y_pred_final['churn'], y_pred_final.final_predicted)
confusion3


# Here, ratio of predicted customers as churned to actually churned is greater when compared to the predicted customers as not churn to actually churned

# In[97]:


# Let's evaluate the other metrics as well

TP = confusion3[1,1] # true positive 
TN = confusion3[0,0] # true negatives
FP = confusion3[0,1] # false positives
FN = confusion3[1,0] # false negatives


# In[98]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[99]:





tempResults = pd.DataFrame({'Method':['Logistic Regression Test'], 'Accuracyscore': [accuracyscore], 'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})

results = pd.concat([results, tempResults])
results = results[['Method','Accuracyscore', 'Sensitivity','Specificity','precision', 'recall']]
results


# ## 2 ) Data Modelling Technique 
# 
# ## Decision treee
# 

# In[100]:


### splitting  data into 70% train set and  30% test set


# In[101]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[102]:


X_train.shape, X_test.shape


# In[103]:


# Decision tree model is run with class weight as balanced due to heavy (8% -92% ) class imbalance of Output variable called 'churn' 


# In[104]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42, max_depth=4, class_weight='balanced')


# In[105]:


dt.fit(X_train, y_train)


# In[106]:


from sklearn.metrics import classification_report

def evaluate_model(classifier):
    print("Train Accuracy :", accuracy_score(y_train, classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, classifier.predict(X_train)))
    print("Train Clasiification report:")
    print(classification_report(y_train, classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, classifier.predict(X_test)))
    print(classification_report(y_test, classifier.predict(X_test)))


# In[107]:


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


# In[108]:


from sklearn import preprocessing, metrics


# In[109]:


from sklearn.metrics import plot_roc_curve

import sklearn.metrics as metrics


plot_roc_curve(dt, X_train, y_train, drop_intermediate=False)
plt.show()


# In[110]:


evaluate_model(dt)


# In[111]:


#Evaluating model performance¶

from sklearn.metrics import confusion_matrix, accuracy_score

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)


# In[112]:


accuracyscore=accuracy_score(y_test, y_test_pred)
print(accuracyscore)
confusion_matrixtrain = confusion_matrix(y_train,y_train_pred)
confusion_matrixtrain


# In[113]:


# Let's evaluate the other metrics as well

TP = confusion_matrixtrain[1,1] # true positive 
TN = confusion_matrixtrain[0,0] # true negatives
FP = confusion_matrixtrain[0,1] # false positives
FN = confusion_matrixtrain[1,0] # false negatives


# In[114]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[115]:


accuracyscore=accuracy_score(y_test, y_test_pred)
print(accuracyscore)
confusion_matrixtest = confusion_matrix(y_test, y_test_pred)
confusion_matrixtest


# In[116]:


# Let's evaluate the other metrics as well

TP = confusion_matrixtest[1,1] # true positive 
TN = confusion_matrixtest[0,0] # true negatives
FP = confusion_matrixtest[0,1] # false positives
FN = confusion_matrixtest[1,0] # false negatives


# In[117]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[118]:


#tempResults = pd.DataFrame({'Method':['Decision Tree Test'], 'Accuracyscore': [accuracyscore], 'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})

#results = pd.concat([results, tempResults])
#results = results[['Method','Accuracyscore', 'Sensitivity','Specificity','precision', 'recall']]
#results


# #### Hyper-parameter tuning for the Decision Tree

# In[119]:


from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')


# In[120]:


params = {
    "max_depth": [2,3,5,10,15,20,25,50],
    "min_samples_leaf": [20,50,75,100,200,500],
    "criterion": ["gini", "entropy"]
}


# In[121]:


grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[122]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[123]:


grid_search.best_score_


# In[124]:


dt_best = grid_search.best_estimator_
dt_best


# In[125]:


plot_roc_curve(dt_best, X_train, y_train)

plt.show()


# In[126]:


evaluate_model(dt_best)


# In[127]:


#Evaluating model performance¶

from sklearn.metrics import confusion_matrix, accuracy_score

y_train_pred = dt_best.predict(X_train)
y_test_pred = dt_best.predict(X_test)


# In[128]:


accuracyscore=accuracy_score(y_test, y_test_pred)
print(accuracyscore)
confusion_matrixtrain = confusion_matrix(y_train,y_train_pred)
confusion_matrixtrain


# In[129]:


# Let's evaluate the other metrics as well

TP = confusion_matrixtrain[1,1] # true positive 
TN = confusion_matrixtrain[0,0] # true negatives
FP = confusion_matrixtrain[0,1] # false positives
FN = confusion_matrixtrain[1,0] # false negatives


# In[130]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[131]:


tempResults = pd.DataFrame({'Method':['Decision Tree Hyperparameter tuning Train'],'Accuracyscore': [accuracyscore], 'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})

results = pd.concat([results, tempResults])
results = results[['Method','Accuracyscore', 'Sensitivity','Specificity','precision', 'recall']]
results


# In[132]:


accuracyscore=accuracy_score(y_test, y_test_pred)
print(accuracyscore)
confusion_matrixtest = confusion_matrix(y_test, y_test_pred)
confusion_matrixtest


# In[133]:


# Let's evaluate the other metrics as well

TP = confusion_matrixtest[1,1] # true positive 
TN = confusion_matrixtest[0,0] # true negatives
FP = confusion_matrixtest[0,1] # false positives
FN = confusion_matrixtest[1,0] # false negatives


# In[134]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[135]:


tempResults = pd.DataFrame({'Method':['Decision Tree Hyperparameter tuning Test'],'Accuracyscore': [accuracyscore], 'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})

results = pd.concat([results, tempResults])
results = results[['Method','Accuracyscore', 'Sensitivity','Specificity','precision', 'recall']]
results


# ### Random Forest

# In[136]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, max_depth=4, max_features=5, random_state=100, 
                            oob_score=True, class_weight='balanced')


# In[137]:


get_ipython().run_cell_magic('time', '', 'rf.fit(X_train, y_train)')


# In[138]:


rf.oob_score_


# In[139]:


plot_roc_curve(rf, X_train, y_train)
plt.show()


# In[140]:


evaluate_model(rf)


# #### Hyper-parameter tuning for the Random Forest

# In[141]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')


# In[142]:


params = {
    'max_depth': [2,3,5,10],
    'min_samples_leaf': [10,20,50,75,100,200],
    'n_estimators': [10, 25, 50, 100],
    'criterion': ['gini','entropy']
}


# In[143]:


grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[144]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[145]:


grid_search.best_score_


# In[146]:


rf_best = grid_search.best_estimator_
rf_best


# In[147]:


plot_roc_curve(rf_best, X_train, y_train)
plt.show()


# In[148]:


evaluate_model(rf_best)


# In[149]:


#Evaluating model performance¶

from sklearn.metrics import confusion_matrix, accuracy_score

y_train_pred = rf_best.predict(X_train)
y_test_pred = rf_best.predict(X_test)


# In[150]:


accuracyscore=accuracy_score(y_test, y_test_pred)
print(accuracyscore)
confusion_matrixtrain = confusion_matrix(y_train,y_train_pred)
confusion_matrixtrain


# In[151]:


# Let's evaluate the other metrics as well

TP = confusion_matrixtrain[1,1] # true positive 
TN = confusion_matrixtrain[0,0] # true negatives
FP = confusion_matrixtrain[0,1] # false positives
FN = confusion_matrixtrain[1,0] # false negatives


# In[152]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[153]:


tempResults = pd.DataFrame({'Method':['Random Forest Hyperparameter tuning Train'],'Accuracyscore': [accuracyscore], 'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})

results = pd.concat([results, tempResults])
results = results[['Method','Accuracyscore', 'Sensitivity','Specificity','precision', 'recall']]
results


# In[154]:


#print(accuracy_score(y_test, y_test_pred))
accuracyscore=accuracy_score(y_test, y_test_pred)
print(accuracyscore)
confusion_matrixtest = confusion_matrix(y_test, y_test_pred)
confusion_matrixtest


# In[155]:


# Let's evaluate the other metrics as well

TP = confusion_matrixtest[1,1] # true positive 
TN = confusion_matrixtest[0,0] # true negatives
FP = confusion_matrixtest[0,1] # false positives
FN = confusion_matrixtest[1,0] # false negatives


# In[156]:


# Calculate the sensitivity, specificity, Precision and recall

sensitivity = round(TP/(TP+FN),2)
specificity = round(TN/(TN+FP),2)
precision = round(TP/(TP+FP),2)
recall = round(TP/(TP+FN),2)

print('sensitivity : {}'.format(sensitivity))
print('specificity : {}'.format(specificity))
print('precision : {}'.format(precision))
print('recall : {}'.format(recall))


# In[157]:


tempResults = pd.DataFrame({'Method':['Random Forest Hyperparameter tuning Test'], 'Accuracyscore': [accuracyscore],'Sensitivity': [sensitivity], 'Specificity': [specificity],'precision': [precision],'recall': [recall]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracyscore','Sensitivity','Specificity','precision', 'recall']]
results


# ### Importance features using random forest

# In[158]:


rf_best.feature_importances_


# In[159]:


imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf_best.feature_importances_
})


# In[160]:


### The top important columns affecting the churn are listed below as per Random forest hyper tuned model


# In[161]:


imp_df.sort_values(by="Imp", ascending=False).head(10)


# In[162]:


### The top important columns affecting the churn are listed below as per Logistic regression


# In[163]:


vif


# # Model Evaluation
# 
# 
# 

# In[164]:


# The performance metrics are of all the 3 models are as shown below " 


# In[165]:


results


# # Best Model
# 
#  The objective of the problem is to identify the features which can help the Telecom  operator to identify the Customers who are planning to churn to other Operater provider. In this category model accuracy is important.
#  
#  At the same time,True churn candidate need to be predicted as Actual churn candidate so that Telecom company can take proactive steps to retain the customer.Hence Sensitivity of the model should be high.
#  
#  We can compromise in case a Negative churn candidate has been falsely predicted as Churn candidate. In that scenario also still company will contact this customer to retain in their network. But this should not be very high otherwise company will incur additional cost. Hence Specificity can be moderate. 
#  
#  Model stability is also important, model should provide similar result in Test and Train.
#  
#  Considering these  parameters, Logistic regression is providing the best result among 3 models with .79 Sensitivity/Recall.
#  
#  Logistic regression is providing consisting Train and Test performance metrics showing model stability.

# # Recommendation
# 
# The top Features for causing Churn from Random forest and Logistic regression are :
# 
# 1. total_ic_mou_8	0.065629
# 2. total_rech_amt_8	0.0556263) 
# 3. arpu_8	0.046496
# 4. total_og_mou_8	0.036674
# 5. max_rech_amt_8	0.028110
# 6. last_day_rch_amt_8	0.023083
# 7. ic_t2m_mou_8	0.021979
# 8. loc_ic_mou_8	0.020844
# 9. loc_og_t2m_mou_8	0.020387
# 10. loc_og_mou_8	0.019819
# 11. last_day_rch_amt_8
# 12. total_rech_num_8
# 13. std_og_mou_8
# 14. vol_2g_mb_8
# 15. spl_ic_mou_8

# In[166]:


# selecting top important features
importantfeatures=['mobile_number','spl_ic_mou_8','vol_2g_mb_8','std_og_mou_8','last_day_rch_amt_8','av_rech_amt_data_8','total_ic_mou_8','total_rech_amt_8','arpu_8','total_og_mou_8','max_rech_amt_8','last_day_rch_amt_8','loc_ic_mou_8','loc_og_t2m_mou_8','loc_og_mou_8','total_rech_num_8','churn']


# In[167]:


high_value_custimportantfeatures=high_value_cust[importantfeatures]


# In[168]:


#Let's see the correlation matrix 
plt.figure(figsize = (20,15))        # Size of the figure
sns.heatmap(high_value_custimportantfeatures.corr(),annot = True,cmap="Blues")
plt.show()


# # Conclusion
# 
# Hence we can see that there is a good amount of positive or negative correlation between the above Features list and output variable Churn
# 
# We can suggest the Telecom company the  strategies based on the dependent factors as shown above:
# 
# 1. total_rech_amnt_8 : Provide Offer on Recharge amount for call to the top customers as this is top factor causing churn
#    We can reduce the overall recharge prepaid amount depending on 3 months, 6months plan instead of monthly plan
#    This will improve the customer satisfaction as recharge amount is reduced and retention period will increase
#    
#   
# 2. total_ic_mou_8 : We can make the incoming call free for these customers 
#  
# 
# 3. average_rech_amt_data_8 : We can provide offer for Data recharge plan in terms of complementary data of 1GB along with Validity for call.
# 
# 

# In[ ]:




