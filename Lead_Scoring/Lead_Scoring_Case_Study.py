#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study
# ### Submitted By - Seema S B

# Problem Statement: 
# 1. To build a model to find out the leads that are most likely to convert into paying customers.
# 2. To find the conversion rate of customers based on scoring of the leads.

# Solution Approach:
# 
# The following approach has been carried out to arrive at the solution for the given problem:
# 1. Data Understanding
# 2. Data Cleaning
# 3. Data Visualization
# 4. Data Preparation
# 5. Modelling

# ### Data Understanding

# #### Import packages

# In[1]:


# Importing the required packages
import warnings
warnings.filterwarnings('ignore')
from math import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


# #### Reading the Data

# In[2]:


# Reading the data from the given csv file
lead_df = pd.read_csv('Leads.csv')


# In[3]:


# Viewing the Data
lead_df.head()


# #### Shape of the dataframe

# In[4]:


# Finding out the no.of rows and columns in the dataset
initial = lead_df.shape
initial


# **Information of the dataframe**

# In[5]:


# To understand the datatypes of each columns and non-null records
lead_df.info()


# We could see that there are many columns which has got `null` values. Also there are various types of columns like `float`, `int64` and `categoricals`.

# ### Data Cleaning

# **Null percentage of columns**

# In[6]:


# Finding out the percentage of null records in each column
round(lead_df.isnull().sum()/len(lead_df) * 100, 2)


# The columns `'Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Asymmetrique Activity Score', 'Asymmetrique Profile Score', 'Tags'` has the maximum percentage of null values in them, so it is better to drop these

# **Dropping the columns with high null percentage**

# In[7]:


# Dropping columns having more null values
lead_df.drop(['Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Asymmetrique Activity Score',
             'Asymmetrique Profile Score', 'Tags'], inplace=True, axis=1)


# **Value Counts of all the columns**

# In[8]:


# Finding the value counts for each columns
for col in lead_df:
    if lead_df[col].dtype == 'O':
        print(round(lead_df[col].value_counts()/len(lead_df) * 100, 2))
        print('=====================================')


# Based on the value counts in each columns, we can infer and perform the below steps:
# 1. The column `Country` has the values mostly as 'India' and also there are null values too. So we should drop this column. The same applies for the column `City`, so dropping it as well
# 2. The columns `'Do Not Email', 'Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content', 'Get updates on DM Content', 'I agree to pay the amount through cheque'` are highly Skewed so dropping them  
# 3. The column `What matters most to you in choosing a course` appears to be highly skewed towards the value `Better Career Prospects`, so it is better to drop the column
# 4. There are various categorical columns which has got the value `Select` as one of its level because it gets populated as default if no other option has been selected by the customer. So this is equivalent to null values
# 5. The columns `Lead Profile` and `How did you hear about X Education` has large number of `Select` which is equivalent to higher null percentage so dropping those columns
# 6. We can drop rows having null values in more than `five` coulmns as it will not impact the target variable
# 7. Though the column `What is your current occupation` has a high number of null records, it is advisable not to drop it because the enrollment of customers to an education platform may be highly impacted based on their occupation or field of work. So only the null records could be dropped while the column is retained 

# **Dropping the columns which doesn't add any information to the data**

# In[9]:


# Dropping columns which doesn't add any information on the dataset
lead_df.drop(['Country', 'City'], axis=1, inplace=True)
round(lead_df.isnull().sum()/len(lead_df) * 100, 2)


# **Dropping the Highly Skewed Columns**

# In[10]:


# Dropping the Highly skewed columns
lead_df.drop(['Do Not Email', 'Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
              'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
              'Update me on Supply Chain Content', 'Get updates on DM Content', 'I agree to pay the amount through cheque', 
              'What matters most to you in choosing a course'], 
             axis=1, inplace=True)


# **Imputing `Select` with `np.nan`**

# In[11]:


# Replacing all the 'Select' values in the dataframe with NULL
lead_df.replace('Select', np.nan, inplace=True)


# **Dropping the columns with high null percentage**

# In[12]:


# Dropping columns with higher null percentage
lead_df.drop(['Lead Profile', 'How did you hear about X Education'], axis=1, inplace=True)


# **Dropping rows having more than five null values**

# In[13]:


# Dropping rows with >5 null values
lead_df.dropna(thresh=lead_df.shape[1]-5, inplace=True)


# **Dropping only null records from potential predictor variable**

# In[14]:


# Dropping only null records instead of the column
lead_df = lead_df[~lead_df['What is your current occupation'].isnull()]


# **Shape and Null Percentage of Columns**

# In[15]:


# Shape of the dataframe
lead_df.shape


# In[16]:


# Null Percentage in coulmns
round(lead_df.isnull().sum()/len(lead_df) * 100, 2)


# **Handling NULL Values in Categorical Columns**

# 1. In column `Lead Source`, the values having its count less than `1.0` are all combined under `Others` as they are less in percentage. The null values are handled by assigning them to `Others`
# 2. In column `Last Activity`, the values having its count less than `1.0` are all combined under `Others` as they are less in percentage. The null values are handled by assigning them to `Others`
# 3. In column `Specialization`, the null values are handled by replacing(imputing) them with `Others`

# In[17]:


# Handling nulls in Lead Source column
def combine(x):
    if x in ('Google','Direct Traffic','Olark Chat', 'Organic Search', 'Reference', 'Welingak Website', 'Referral Sites'):
        return x
    else:
        return 'Others'

lead_df['Lead Source'].replace('google', 'Google', inplace=True)
lead_df['Lead Source'] = lead_df['Lead Source'].apply(combine)
round(lead_df['Lead Source'].value_counts()/len(lead_df) * 100, 2)


# In[18]:


# Handling nulls in Last Activity column
def combine(x):
    if x in ('Email Opened','SMS Sent','Olark Chat Conversation','Page Visited on Website','Converted to Lead','Email Bounced',
            'Email Link Clicked','Form Submitted on Website','Unreachable'):
        return x
    else:
        return 'Others'

lead_df['Last Activity'] = lead_df['Last Activity'].apply(combine)
round(lead_df['Last Activity'].value_counts()/len(lead_df) * 100, 2)


# In[19]:


# Handling nulls in Specialization column
lead_df['Specialization'].replace(np.nan, 'Others', inplace=True)
round(lead_df['Specialization'].value_counts()/len(lead_df) * 100, 2)


# **Shape and Null Percentage of Columns**

# In[20]:


lead_df.shape


# In[21]:


round(lead_df.isnull().sum()/len(lead_df)*100, 2)


# **Mapping 'Yes' and 'No' to 1 and 0 respectively in column `'A free copy of Mastering The Interview'`**

# In[22]:


# Converting the binomial categorical column to numerical
lead_df['A free copy of Mastering The Interview'] = lead_df['A free copy of Mastering The Interview'].map({'Yes': 1, 'No': 0})


# **Handling NULL Values in numerical variables**

# In[23]:


# Imputing the null values in Total Visits column with its median value
lead_df['TotalVisits'].replace(np.nan, lead_df['TotalVisits'].median(), inplace=True)

# Imputing the null values in Page Views Per Visit column with its median value
lead_df['Page Views Per Visit'].replace(np.nan, lead_df['Page Views Per Visit'].median(), inplace=True)


# In[24]:


# Dropping the unique id column too
lead_df.drop('Prospect ID', axis=1, inplace=True)


# **NULL Percentage after Data Cleaning**

# In[25]:


# Percentage of null values left after data cleaning
round(lead_df.isnull().sum()/len(lead_df)*100, 2)


# All the columns are now non-null

# In[26]:


# Viewing the top five rows
lead_df.head()


# In[27]:


# Shape of the dataframe
lead_df.shape


# In[28]:


# Information on the columns
lead_df.info()


# **Percentage of data retained**

# In[29]:


# The percentage of data retained from the initial dataset
len(lead_df)/initial[0]*100


# - We have 70.88% of rows which is quite enough for analysis

# ### Data Visualization

# #### Univariate Analysis

# In[30]:


# Plotting the numerical variables
plt.figure(figsize=(14,10))
plt.subplot(2,3,1)
sns.boxplot(lead_df['Total Time Spent on Website'])
plt.subplot(2,3,2)
sns.boxplot(lead_df['TotalVisits'])
plt.subplot(2,3,3)
sns.boxplot(lead_df['Page Views Per Visit'])
plt.show()


# The columns `TotalVisits` and `Page Views Per Visit` have outliers in it and needs to be treated

# **Handling the Outliers**

# In[31]:


# Capping the outliers to its 99th quantile value in Total Visits column
quant = lead_df['TotalVisits'].quantile([0.99])
lead_df['TotalVisits'] = np.clip(lead_df['TotalVisits'],
                                 lead_df['TotalVisits'].quantile([0.01,0.99][0]),
                                 lead_df['TotalVisits'].quantile([0.01,0.99][1]))


# In[32]:


sns.boxplot(lead_df['TotalVisits'])


# In[33]:


# Capping the outliers to its 99th quantile value in Page Views Per Visit column
quant = lead_df['Page Views Per Visit'].quantile([0.99])
lead_df['Page Views Per Visit'] = np.clip(lead_df['Page Views Per Visit'],
                                          lead_df['Page Views Per Visit'].quantile([0.01,0.99][0]), 
                                          lead_df['Page Views Per Visit'].quantile([0.01,0.99][1]))


# In[34]:


sns.boxplot(lead_df['Page Views Per Visit'])


# In[35]:


# Correlation Matrix - Heatmap
sns.heatmap(lead_df.corr(), annot=True)


# From the heat map we can see that the columns `TotalVisits` and `Page Views Per Visit` are highly correlated.

# ### Data Preparation

# #### Dummy Variables Creation for Categorical variables

# In[36]:


# Creating dummy variables for the column Lead Origin
Lead_Origin = pd.get_dummies(lead_df['Lead Origin'], drop_first=True)
Lead_Origin.head()


# In[37]:


# Creating dummy variables for the column Lead Source
Lead_Source = pd.get_dummies(lead_df['Lead Source'])
Lead_Source.drop('Others', axis=1, inplace=True)
Lead_Source.head()


# In[38]:


# Creating dummy variables for the column Lead Activity
Last_Activity = pd.get_dummies(lead_df['Last Activity'], prefix='Last')
Last_Activity.drop('Last_Others', axis=1, inplace=True)
Last_Activity.head()


# In[39]:


# Creating dummy variables for the column Specialization
Specialization = pd.get_dummies(lead_df['Specialization'])
Specialization.drop('Others', axis=1, inplace=True)
Specialization.head()


# In[40]:


# Creating dummy variables for the column 'What is your current occupation'
occupation = pd.get_dummies(lead_df['What is your current occupation'])
occupation.drop('Other', axis=1, inplace=True)
occupation.head()


# In[41]:


# Creating dummy variables for the column 'Last Notable Activity'
activity = pd.get_dummies(lead_df['Last Notable Activity'])
activity.drop('Unsubscribed', axis=1, inplace=True)
activity.head()


# In[42]:


# Concatenating all the dummy variables created to the main dataframe
leads = pd.concat([lead_df,Lead_Origin,Lead_Source,occupation,Last_Activity,Specialization,activity], axis=1)
leads.head()


# In[43]:


# Dropping off the original columns
leads.drop(['Lead Origin','Lead Source','What is your current occupation','Last Activity','Specialization'
            ,'Last Notable Activity'], axis=1, inplace=True)
leads.head()


# In[44]:


# Number of rows and columns
leads.shape


# **Correlation Matrix [HeatMap]**

# In[45]:


# Plotting the correlation matrix for the dataset
plt.figure(figsize=(44,40))
sns.heatmap(round(leads.corr(),2), annot=True)


# In[46]:


# Top 10 highly correlated variables
leads.corr().unstack().sort_values(ascending=False).drop_duplicates().head(10)


# In[47]:


# Top 10 inversely correlated variables
leads.corr().unstack().sort_values(ascending=True).drop_duplicates().head(10)


# #### Train test split and scaling of data

# In[48]:


# Splitting the dataframe into train and test
np.random.seed(0)
df_train, df_test = train_test_split(leads, train_size=0.7, test_size=0.3, random_state=100)


# In[49]:


# Scaling the data to bring all in one scale
scaler = StandardScaler()


# In[50]:


# Apply the scaling metrics
num_vars = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[51]:


# Predictor and Target Variable Split
y_train = df_train.pop('Converted')
X_train = df_train.drop('Lead Number', axis=1)


# ### Modelling 

# - Let's now move to model building. As you can see that there are a lot of variables present in the dataset which we cannot deal with. So the best way to approach this is to select a small set of features from this pool of variables using RFE.

# **RFE Approach**

# In[52]:


# Performing RFE approach for 15 variables
logreg = LogisticRegression()
rfe = RFE(logreg, 15)         # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[53]:


# RFE result on all variables
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[54]:


# Selected 15
col = X_train.columns[rfe.support_]
col


# - Now you have all the variables selected by RFE and to check the statistics part, i.e. the p-values and the VIFs, let's use these variables to create a logistic regression model using statsmodels.

# **Logistic regression using statsmodels [Model 1]**

# In[55]:


# Performing logistic regression using stats models
X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial()).fit()
logm1.summary()


# In[56]:


# Calculating the VIF values
VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# The VIF Value for all the columns are less than 5. But there are few columns which has high p-values. So based on high p-value, the column `Had a Phone Conversation` should be dropped.

# In[57]:


# Dropping the column with high p-value
col = col.drop('Had a Phone Conversation')
col


# **Logistic regression using statsmodels [Model 2]**

# In[58]:


# Performing logistic regression using stats models
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial()).fit()
logm2.summary()


# In[59]:


# Calculating the VIF values
VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# The VIF Value for all the columns are less than 5. But there are few columns which has high p-values. So based on high p-value, the column `Housewife` should be dropped.

# In[60]:


# Dropping the column with high p-value
col = col.drop('Housewife')
col


# **Logistic regression using statsmodels [Model 3]**

# In[61]:


# Performing logistic regression using stats models
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial()).fit()
logm3.summary()


# In[62]:


# Calculating the VIF values
VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# Since all the p-values and VIF values are low for all the columns, let's consider the `Model 3` as the final model

# ### Model Evaluation

# In[63]:


# Predicting the target values based on the predictor values using the final model
y_train_pred = logm3.predict(X_train_sm)

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Actual vs Predicted

# In[64]:


# Creating a dataframe with the actual conversion flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final.head()


# #### Random Cut-off for Predicted

# In[65]:


# Creating new column 'Predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['Predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# #### Evaluate the model using Confusion Matrix

# In[66]:


# Plot the confusion matrix
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
confusion


# In[1]:


## Predicted       Not Converted     Converted
## Actual
## Not Converted        1983            395
## Converted             563           1644


# In[67]:


# Calculating the accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[68]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[69]:


# Calculate the sensitivity

TP/(TP+FN)


# In[70]:


# Calculate the specificity

TN/(TN+FP)


# For the chosen random cut-off probability value, the Sensitivity is lower compared to specificity which clearly suggests that it's not the optimal cut-off. Let's check the ROC curve for the same.

# **ROC Curve**

# In[71]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# In[72]:


# Plotting the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, 
                                         drop_intermediate = False)

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# The area under the ROC curve is 0.87 which indicates that the chosen cut-off may not be optimal and some better cut-off could be used

# **Optimal Cut-Off Selection**

# In[73]:


# Creating columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[74]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs

cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])

num = [float(x)/10 for x in range(10)]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[75]:


# Plotting the probability value across accuracy, sensitivity and specificity

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# As you can see that around 0.38, you get the optimal values of the three metrics. So let's choose `0.38` as our cutoff now.

# **Predicting with the Optimal cut-off**

# In[76]:


# The predictions are done with the new optimal cut-off value
y_train_pred_final['Final_Predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.38 else 0)
y_train_pred_final.head()


# In[77]:


# Confusion Matrix
confusion_train = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted)
confusion_train


# In[2]:


## Predicted       Not Converted     Converted
## Actual
## Not Converted        1691            687
## Converted             317           1890


# In[78]:


# Calculating the accuracy score
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# In[79]:


# Let's evaluate the other metrics as well

TP = confusion_train[1,1] # true positive 
TN = confusion_train[0,0] # true negatives
FP = confusion_train[0,1] # false positives
FN = confusion_train[1,0] # false negatives


# In[80]:


# Calculate the sensitivity

TP/(TP+FN)


# In[81]:


# Calculate the specificity

TN/(TN+FP)


# In[82]:


# calculate precision

TP/(TP+FP)


# In[83]:


# calculate F1-Score

metrics.f1_score(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted)


# With the chosen optimal cut-off, `0.38`:
# 1. Sensitivity is high compared to specificity
# 2. Precision is also high
# 3. The F1-Score is almost 79% which means the model has a better performance

# ### Making prediction on Test set

# #### Scaling the test dataset

# In[84]:


# Popping out the target variable from the test data
y_test = df_test.pop('Converted')

# Scaling the test using the same metrics used for train
df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[85]:


# Scaled Test Predictor variables
X_test = df_test[col]
X_test.head()


# #### Predicting on the test dataset

# In[86]:


# Predicting on test dataset using the final model
X_test_sm = sm.add_constant(X_test)
y_test_pred = logm3.predict(X_test_sm)
y_test_pred.head()


# In[87]:


# Creating a dataframe with the predicted probabilities
y_test_pred = pd.DataFrame(y_test_pred, columns=['Converted_Prob'], dtype='float64')
y_test_pred.head()


# In[88]:


# Creating a dataframe with the target values
y_test = pd.DataFrame(y_test, columns=['Converted'], dtype='int64')
y_test.head()


# In[89]:


# Creating new datafram with target variable and predicted variable
y_test_final = pd.concat([y_test, y_test_pred], axis=1)
y_test_final.head()


# **Predicting the Test data using the optimal cut-off chosen**

# In[90]:


# Creating new column 'Predicted' with 1 if Converted_Prob > 0.38 else 0
y_test_final['Predicted'] = y_test_final.Converted_Prob.map(lambda x: 1 if x > 0.38 else 0)
y_test_final.head()


# #### Evaluate the model using Confusion Matrix

# In[91]:


# Confusion Matrix
confusion_test = metrics.confusion_matrix(y_test_final.Converted, y_test_final.Predicted)
confusion_test


# In[3]:


## Predicted       Not Converted     Converted
## Actual
## Not Converted         697            284
## Converted             160            824


# In[92]:


# Calculating the Accuracy
metrics.accuracy_score(y_test_final.Converted, y_test_final.Predicted)


# In[93]:


# Let's evaluate the other metrics as well

TP = confusion_test[1,1] # true positive 
TN = confusion_test[0,0] # true negatives
FP = confusion_test[0,1] # false positives
FN = confusion_test[1,0] # false negatives


# In[94]:


# Calculate the sensitivity

TP/(TP+FN)


# In[95]:


# Calculate the specificity

TN/(TN+FP)


# In[96]:


# calculate precision

TP/(TP+FP)


# In[97]:


# calculate F1-Score

metrics.f1_score(y_test_final.Converted, y_test_final.Predicted)


# For the test data:
# 1. The Accuracy is high
# 2. Sensitivity is high compared to specificity
# 3. Precision is also pretty high
# 4. F1-score is about 78%
# 
# Overall the final model looks good after evaluation

# ### Precision and Recall

# #### On train dataset

# In[98]:


# Confusion Matrix
confusion_train = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Predicted)
confusion_train


# In[4]:


## Predicted       Not Converted     Converted
## Actual
## Not Converted        1691            687
## Converted             317           1890


# In[99]:


# calculate precision
# TP/(TP+FP)

confusion_train[1,1]/(confusion_train[1,1] + confusion_train[0,1])


# In[100]:


# calculate recall
# TP/(TP+FN)

confusion_train[1,1]/(confusion_train[1,1] + confusion_train[1,0])


# Precision is less when compared to Recall

# #### Precision and Recall tradeoff

# In[101]:


p, r, thres = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# In[102]:


plt.plot(thres, p[:-1], "g-")
plt.plot(thres, r[:-1], "r-")
plt.show()


# Based on the trade-off value 0.4 is chosen as the threshold for final prediction

# **Making Final Predictions using `0.4` as the cut-off value**

# In[103]:


# Creating new column 'Final_Pred_PR' with 1 if Converted_Prob > 0.4 else 0 and evaluating the model
y_train_pred_final['Final_Pred_PR'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.4 else 0)
y_train_pred_final.head()


# In[104]:


# Confusion Matrix
confusion_pr = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Pred_PR)
confusion_pr


# In[105]:


# Calculating the accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Pred_PR)


# In[106]:


# Let's evaluate the other metrics as well

TP = confusion_pr[1,1] # true positive 
TN = confusion_pr[0,0] # true negatives
FP = confusion_pr[0,1] # false positives
FN = confusion_pr[1,0] # false negatives


# In[107]:


# Calculate the recall

TP/(TP+FN)


# In[108]:


# calculate precision

TP/(TP+FP)


# In[109]:


# calculate F1-Score

metrics.f1_score(y_test_final.Converted, y_test_final.Predicted)


# #### Prediction on the test dataset

# In[110]:


# Creating new column 'Predicted_PR' with 1 if Converted_Prob > 0.4 else 0 and evaluating the model
y_test_final['Predicted_PR'] = y_test_final.Converted_Prob.map(lambda x: 1 if x > 0.4 else 0)
y_test_final.head()


# In[111]:


# Confusion Matrix
confusion_test_pr = metrics.confusion_matrix(y_test_final.Converted, y_test_final.Predicted_PR)
confusion_test_pr


# In[5]:


## Predicted       Not Converted     Converted
## Actual
## Not Converted         751            230
## Converted             205            779


# In[112]:


# Calculating the accuracy
metrics.accuracy_score(y_test_final.Converted, y_test_final.Predicted_PR)


# In[113]:


# Let's evaluate the other metrics as well

TP = confusion_test_pr[1,1] # true positive 
TN = confusion_test_pr[0,0] # true negatives
FP = confusion_test_pr[0,1] # false positives
FN = confusion_test_pr[1,0] # false negatives


# In[114]:


# Calculate the recall

TP/(TP+FN)


# In[115]:


# calculate precision

TP/(TP+FP)


# In[116]:


# calculate F1-Score

metrics.f1_score(y_test_final.Converted, y_test_final.Predicted)


# Based on the predictions:
# 1. Precision is low compared to the Recall
# 2. F1-Score is around 78%

# ### Assigning Lead Scores on test dataset

# In[117]:


# Lead Score function
def score(x):
    return int(100-74+(7*log(x/(1-x))/log(2)))

y_test_final['Score'] = y_test_final.Converted_Prob.apply(score)


# In[118]:


y_test_final.head()


# In[119]:


# Assign the score to 0 for those having negative scores
y_test_final.Score[y_test_final.Score < 0] = 0


# In[120]:


# Top 5 Leads based on their lead score
y_test_final.sort_values(by='Score', ascending=False).head()


# In[121]:


# Concatenating X and y to find out the details about the leads using their lead scores
lead_score = pd.concat([X_test_sm, y_test_final], axis=1)


# In[122]:


# Top 5 Leads based on their lead score
lead_score.sort_values(by='Score', ascending=False).head()


# In[123]:


# Least 5 Leads based on their lead score
lead_score.sort_values(by='Score').head()


# **Hot Leads**

# In[124]:


# The leads with score greater than 70 will give us the hot leads
y_test_final[y_test_final.Score > 70].sort_values(by='Score').count()


# **Converted Leads even when they weren't reached on Phone**

# In[127]:


lead_score[lead_score.Unreachable == 1]


# **Summary**

# - The equation for Logistic Regression based on the final model `Model 3` is:
#  $ Converted = 0.9303 \times const + 1.1188 \times Total Time Spent on Website + 3.2419 \times Lead Add Form
# -0.6192 \times Direct Traffic + 1.2087 \times Olark Chat + 2.5373 \times Welingak Website	
# -1.5277 \times Student	-1.5044 \times Unemployed + 1.0724 \times Working Professional
# -1.3279 \times Last\_Email Bounced -0.8449 \times Last\_Olark Chat Conversation
# +0.9830 \times Last\_SMS Sent -0.8698 \times Modified + 2.5536 \times Unreachable $

# - The Optimal cut-off chosen is `0.38`
# 
# - Model Accuracy:`0.7740458015267175`
# 
# - Sensitivty:`0.8373983739837398`
# 
# - Specificity:`0.7104994903160041`
# 
# - F1-Score:`0.7877629063097515`

# - After Precision-Recall Tradeoff, the cut-off for probability is `0.4`
# 
# - Model Accuracy:`0.7786259541984732`
# 
# - Precision:`0.7720515361744301`
# 
# - Recall:`0.7916666666666666`
# 
# - F1-Score:`0.7877629063097515`
