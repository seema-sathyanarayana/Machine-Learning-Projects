#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement :
# 
# Evanston Hospital is a comprehensive acute-care facility in Illinois, US. The hospital offers a wide range of services and surgical specialities, in addition to having high-end lab capabilities. Despite spending a considerable amount of resources on improving its services, the hospital’s CMS rating has remained at 3 for the past 5 years, and this has led to a steady decline in revenue for the hospital. For hospitals like Evanston, these ratings directly influence the choice made by consumers who are looking for a healthcare provider and would, therefore, have a significant impact on the hospitals’ revenues. As a consulting company hired by Evanston, our task is to identify possible root causes for the hospital getting such an average rating and recommend measures to mitigate this problem.

# ### Solution Approach:
# 
# The following approach has been carried out to arrive at the solution for the given problem:
# 
# 1. Data Understanding
# 2. Data Cleaning
# 3. Data Visualization
# 4. Data Preparation
# 5. Modelling

# ### 1. Data Understanding

# ###### Import the packages

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import plot_confusion_matrix, classification_report


# ###### Read the data from csv file

# In[2]:


genInfo = pd.read_csv('Hospital General Information.csv', encoding='cp1252')
genInfo.head()


# ###### Checking number of rows and columns in dataset

# In[3]:


genInfo.shape


# ###### Detailed information of dataset

# In[4]:


genInfo.info()


# ###### Statistical description of dataset

# In[5]:


genInfo.describe(include='object')


# - We could see that there are many columns which has got `Null` values. Also there are both types of columns `numeric` and `categoricals`.

# ### 2. Data Cleaning

# ###### Checking for percentage of Null values in each columns

# In[6]:


round(100*genInfo.isna().sum()/len(genInfo), 2)


# - As there few columns with more than `50%` null values, such columns are dropped. The columns that are dropped are: `Hospital overall rating footnote`, `Safety of care national comparison`, `Readmission national comparison`, `Patient experience national comparison`, `Effectiveness of care national comparison`, `Timeliness of care national comparison`, `Efficient use of medical imaging national comparison`.

# ###### Dropping the columns with more than 50% of null value

# In[7]:


genInfo.drop(genInfo.columns[round(100*genInfo.isna().sum()/len(genInfo), 2) > 50], axis=1, inplace=True)


# ###### Checking the number of rows and columns left

# In[8]:


genInfo.shape


# ###### Imputing the missing values with 'N' in column 'Meets criteria for meaningful use of EHRs'

# In[9]:


genInfo['Meets criteria for meaningful use of EHRs'].fillna('N', inplace=True)


# ###### Checking for percentage of Null values in each column

# In[10]:


round(100*genInfo.isna().sum()/len(genInfo), 2)


# ###### Finding the value counts for each columns

# In[11]:


for col in genInfo:
    if genInfo[col].dtype == 'O':
        print(round(genInfo[col].value_counts()/len(genInfo) * 100, 2))
        print('='*100, end='\n\n')


# ###### Replacing the 'Not Available' value with '0' in the column 'Hospital overall rating'
# Assuming that 'Not Available' as 0 in column Hospital overall rating for further analysis

# In[12]:


genInfo['Hospital overall rating'].replace('Not Available', 0, inplace=True)


# ###### Converting the data type of 'Hospital overall rating' column to 'integer'

# In[13]:


genInfo['Hospital overall rating'] = genInfo['Hospital overall rating'].astype('int64')


# ###### Imputing the values in 'Emergency Services', 'Meets criteria for meaningful use of EHRs' columns with 0s and 1s instead of 'No' and 'Yes' respectively

# In[14]:


binvars = ['Emergency Services', 'Meets criteria for meaningful use of EHRs']

def bin_vars(x):
    return x.map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0})

genInfo[binvars] = genInfo[binvars].apply(bin_vars)
genInfo.head()


# In[15]:


genInfo.info()


# ### 3. Data Visualization

# ###### Number of Hospitals under each type

# In[16]:


plt.figure(figsize=(8,4))
sns.countplot(genInfo['Hospital Type'])
plt.show()


# - From above graph, it can be seen that `most` of the Hospitals fall under the type `Acute Care` and there are `very less` Hospitals which has facility for `Childrens`.

# ###### Number of Hospitals under each ownership

# In[17]:


plt.figure(figsize=(8,4))
sns.countplot(genInfo['Hospital Ownership'])
plt.xticks(rotation=90)
plt.show()


# - From the graph, Hospitals under the ownership of `Voluntary non-profit - Private` are `very high` when compared to others. `Tribal` owns `very few` hospitals. Also hospitals under Physician, Government-State and Government-Federal ownership are less.

# ###### Number of Hospitals under each Rating

# In[18]:


plt.figure(figsize=(8,4))
sns.countplot(genInfo['Hospital overall rating'])
plt.xticks(rotation=90)
plt.show()


# - Most of the hospitals fall under `Rating-3` and also rating of many hospitals ratings are `not available`. There are `very few` hospitals which are having `Rating-1`, `Rating-5`.

# ### 4. Data Preparation

# ###### Creating dummies for Categorial columns

# In[19]:


d1 = pd.get_dummies(genInfo['Hospital Type'], prefix='Type')
d1.drop('Type_Childrens', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Hospital Ownership'], prefix='Ownership')
d1.drop('Ownership_Tribal', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Mortality national comparison'], prefix='Mortality')
d1.drop('Mortality_Below the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Safety of care national comparison'], prefix='Safety')
d1.drop('Safety_Below the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Readmission national comparison'], prefix='Readmission')
d1.drop('Readmission_Above the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Patient experience national comparison'], prefix='Patient')
d1.drop('Patient_Below the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Effectiveness of care national comparison'], prefix='Effectiveness')
d1.drop('Effectiveness_Below the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Timeliness of care national comparison'], prefix='Timeliness')
d1.drop('Timeliness_Below the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)

d1 = pd.get_dummies(genInfo['Efficient use of medical imaging national comparison'], prefix='Efficient')
d1.drop('Efficient_Above the National average', axis=1, inplace=True)
genInfo = pd.concat([genInfo,d1], axis=1)


# In[20]:


# Viewing the dataset after dummies are created

genInfo.head()


# ###### Dropping the columns for which dummy columns were created

# In[21]:


genInfo.drop(['Hospital Type', 'Hospital Ownership', 'Mortality national comparison', 'Safety of care national comparison',
              'Readmission national comparison', 'Patient experience national comparison',
              'Effectiveness of care national comparison', 'Timeliness of care national comparison',
              'Efficient use of medical imaging national comparison'], axis=1, inplace=True)


# In[22]:


# Viewing the dataset

genInfo.head()


# In[23]:


# Number of rows and columns in dataset

genInfo.shape


# ###### Plotting the correlation matrix for the dataset

# In[24]:


plt.figure(figsize=(44,40))
sns.heatmap(round(genInfo.corr(),2), annot=True, cmap='Reds')
plt.show()


# - There are few variables which are positively correlated to each other and some are negatively correlated as well.

# ###### Dropping other categorical columns like City, Country, Phone Number etc which don't add much value to dataset as we need to find the factors influencing the rating of Hospital

# In[25]:


df = genInfo.drop(['Provider ID', 'Hospital Name', 'Address', 'City', 'State', 'ZIP Code', 'County Name', 'Phone Number'],axis=1)


# In[26]:


df.head()


# In[27]:


# Information of dataset used for modelling

df.info()


# ### 5. Model Building
# 
# ###    Logistic Regression

# ###### Splitting the dataframe into train and test

# In[28]:


X = df.drop('Hospital overall rating', 1)
y = df['Hospital overall rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[29]:


X_train.shape, X_test.shape


# ###### Create a LogisticRegression object and performing RFE to top 20 features
# Let's now move to model building. As you can see that there are a lot of variables present in the dataset which we cannot deal with. So the best way to approach this is to select a small set of features from this pool of variables using RFE.

# In[30]:


logreg = LogisticRegression(class_weight='balanced', multi_class='multinomial')


# In[31]:


# Selecting 20 variables

rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train)


# In[32]:


# Let's take a look at which features have been selected by RFE

list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[33]:


# Put all the columns selected by RFE in the variable 'col'

col = X_train.columns[rfe.support_]
col


# ###### Fit a logistic Regression model on X_train after adding a constant and output the summary and Calculating VIF value

# ###### Logit Model 1

# In[34]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train, X_train_sm)
res = logm1.fit()
res.summary()


# In[35]:


# Calculating the VIF values

VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# In[36]:


col = col.drop('Mortality_Not Available')
col


# ###### Logit Model 2

# In[37]:


# Fit a logistic Regression model on X_train

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train, X_train_sm)
res = logm2.fit()
res.summary()


# In[38]:


# Calculating the VIF values

VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# In[39]:


col = col.drop('Timeliness_Not Available')
col


# ###### Logit Model 3

# In[40]:


# Fit a logistic Regression model on X_train

X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train, X_train_sm)
res = logm3.fit()
res.summary()


# In[41]:


# Calculating the VIF values

VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# In[42]:


col = col.drop('Emergency Services')
col


# ###### Logit Model 4

# In[43]:


# Fit a logistic Regression model on X_train

X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train, X_train_sm)
res = logm4.fit()
res.summary()


# In[44]:


# Calculating the VIF values

VIF = pd.DataFrame()
VIF['Features'] = X_train[col].columns
VIF['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
VIF['VIF'] = round(VIF['VIF'], 2)
VIF = VIF.sort_values(by = 'VIF', ascending = False)
VIF


# - Since all the p-values and VIF values are low for all the columns, let's consider the `Model 4` as the final model

# #### Model Evaluation

# In[45]:


# Predicting the target values based on the predictor values using the final model

y_train_pred = res.predict(X_train_sm)

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# #### Actual vs Predicted

# In[46]:


# Create a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Rating':y_train.values, 'Rating_Prob':y_train_pred})
y_train_pred_final.head()


# In[47]:


# Creating new column 'Predicted' by rounding-off the value to next highest number

y_train_pred_final['Predicted'] = y_train_pred_final.Rating_Prob.map(lambda x: round(x) if x > 0 else 0)

# Let's see the head

y_train_pred_final.head()


# In[48]:


# Create confusion matrix 

confusion = confusion_matrix(y_train_pred_final.Rating, y_train_pred_final.Predicted)
print(confusion)


# In[49]:


# Let's check the overall accuracy

accuracyscore = round(accuracy_score(y_train_pred_final.Rating, y_train_pred_final.Predicted), 2)
print(accuracyscore)


# In[50]:


# Printing classification report

print(classification_report(y_train_pred_final.Rating, y_train_pred_final.Predicted))


# In[51]:


# Adding results into the dataframe

results = pd.DataFrame({'Method':['Logistic Regression Train'], 'Accuracy': [accuracyscore]})

results = results[['Method','Accuracy']]
results


# ###### Making prediction on Test dataset

# In[52]:


# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[col]


# In[53]:


# Add a constant to X_test

X_test_sm = sm.add_constant(X_test_new)


# In[54]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

y_test_pred = res.predict(X_test_sm)


# In[55]:


y_test_pred.head()


# In[56]:


# Converting y_pred to a dataframe

y_pred = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 

y_pred.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[57]:


# Append y_test_df and y_pred

y_pred_final = pd.concat([y_test_df, y_pred],axis=1)

# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Rating_Prob'})
y_pred_final.head()


# In[58]:


# Creating new column 'Predicted' by rounding-off the value to next highest number

y_pred_final['Predicted'] = y_pred_final.Rating_Prob.map(lambda x: round(x) if x > 0 else 0)

# Let's see the head
y_pred_final.head()


# In[59]:


# Create confusion matrix 

confusion = confusion_matrix(y_pred_final['Hospital overall rating'], y_pred_final.Predicted)
print(confusion)


# In[60]:


# Let's check the overall accuracy

accuracyscore = round(accuracy_score(y_pred_final['Hospital overall rating'], y_pred_final.Predicted), 2)
print(accuracyscore)


# In[61]:


# Printing classification report

print(classification_report(y_pred_final['Hospital overall rating'], y_pred_final.Predicted))


# In[62]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Logistic Regression Test'], 'Accuracy': [accuracyscore]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# ### Decision Tree

# In[63]:


# splitting  data into 70% train set and  30% test set

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[64]:


# Number of rows and columns in Train and Test dataset

X_train.shape, X_test.shape


# In[65]:


# Creating object of Decision tree classifier and fitting it

dt = DecisionTreeClassifier(random_state=42, max_depth=4, class_weight='balanced')
dt.fit(X_train, y_train)


# In[66]:


# Creating a function called 'evaluate_model' to find accuracy and other evaluation metrics

def evaluate_model(classifier):
    accuracy_train = round(accuracy_score(y_train, classifier.predict(X_train)), 2)
    print("Train Accuracy :", accuracy_train)
    print("Train Confusion Matrix:")
    print(plot_confusion_matrix(classifier, X_train, y_train, cmap=plt.cm.Blues))
    plt.show()
    print("Train Clasification report:")
    print(classification_report(y_train, classifier.predict(X_train)))
    print("-"*50)
    accuracy_test = round(accuracy_score(y_test, classifier.predict(X_test)), 2)
    print("Test Accuracy :", accuracy_test)
    print("Test Confusion Matrix:")
    print(plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues))
    plt.show()
    print("Test Clasification report:")
    print(classification_report(y_test, classifier.predict(X_test)))
    return accuracy_train, accuracy_test


# ###### Evaluating the model and printing the results

# In[67]:


accuracy_train, accuracy_test = evaluate_model(dt)


# In[68]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Decision Tree Train'], 'Accuracy': [accuracy_train]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# In[69]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Decision Tree Test'], 'Accuracy': [accuracy_test]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# ###### Hyper-parameter tuning for the Decision Tree

# In[70]:


dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')


# In[71]:


params = {
    "max_depth": [2,3,5,10],
    "min_samples_leaf": [5,10,20,50,75,100],
    "criterion": ["gini", "entropy"]
}


# In[72]:


grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[73]:


grid_search.fit(X_train, y_train)


# In[74]:


# Printing the best score of the estimator

round(grid_search.best_score_, 2)


# In[75]:


# Printing the best estimator

dt_best = grid_search.best_estimator_
dt_best


# ###### Evaluating the model and printing the results

# In[76]:


accuracy_train, accuracy_test = evaluate_model(dt_best)


# In[77]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Decision Tree Hyperparameter Tuning Train'], 'Accuracy': [accuracy_train]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# In[78]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Decision Tree Hyperparameter Tuning Test'], 'Accuracy': [accuracy_test]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# ### Random Forest

# In[79]:


# Creating the object of Random forest classifier and fitting it

rf = RandomForestClassifier(n_estimators=10, max_depth=4, max_features=5, random_state=100, 
                            oob_score=True, class_weight='balanced_subsample')


# In[80]:


rf.fit(X_train, y_train)


# In[81]:


# Printing the OOB score

round(rf.oob_score_, 2)


# ###### Evaluating the model and printing the results

# In[82]:


accuracy_train, accuracy_test = evaluate_model(rf)


# In[83]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Random Forest Train'], 'Accuracy': [accuracy_train]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# In[84]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Random Forest Test'], 'Accuracy': [accuracy_test]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# ###### Hyper-parameter tuning for the Random Forest

# In[85]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', oob_score=True)


# In[86]:


params = {
    'max_depth': [2,3,5,10],
    'min_samples_leaf': [10,20,50,75,100,200],
    'n_estimators': [10, 25, 50, 100],
    'criterion': ['gini','entropy']
}


# In[87]:


grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[88]:


grid_search.fit(X_train, y_train)


# In[89]:


# Printing the best score of the estimator

round(grid_search.best_score_, 2)


# In[90]:


# Printing the best estimator

rf_best = grid_search.best_estimator_
rf_best


# ###### Evaluating the model and printing the results

# In[91]:


accuracy_train, accuracy_test = evaluate_model(rf_best)


# In[92]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Random Forest Hyperparameter Tuning Train'], 'Accuracy': [accuracy_train]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# In[93]:


# Adding results into the dataframe

tempResults = pd.DataFrame({'Method':['Random Forest Hyperparameter Tuning Test'], 'Accuracy': [accuracy_test]})

results = pd.concat([results, tempResults])
results = results[['Method', 'Accuracy']]
results


# ### Importance features using Random forest

# In[94]:


rf_best.feature_importances_


# In[95]:


imp_df = pd.DataFrame({
    "VarName": X_train.columns,
    "Imp": rf_best.feature_importances_
})


# In[96]:


imp_df.sort_values(by="Imp", ascending=False).head(10)


# ### Result and Explanation

# - The problem statement is to build a model which predicts the hospital rating and also to find the factors that impact the rating of hospital.
# - Based on the table below, `Decision Tree with hyperparameter tuning` is the `Best Model` for rating prediction of past hospital data.
# - Decision tree with hyperparameter tuning is chosen because the accuracy is pretty good and also the `Recall` value is better compared to other models. 
# - Even from the confusion matrix, it can be seen that the probability of Actual rating predicted correctly by the model is good when compared to others which is important for a Hospital. Since the patients opting for a particular hospital depends on the rating.

# ![Result_table.JPG](attachment:Result_table.JPG)

# The **`Important features`** that impact the Hospital overall rating are:
# 1. `Patient experience` - needs to be above the national average i.e., hospital need to make sure they provide great experience for their patients.
# 2. `Readmission` - needs to be below the national average i.e., hospital need to make sure it gives proper and required treatment to patients to avoid readmission.
# 3. `Effectiveness of care` - Readmission and effectiveness of care goes hand in hand. If treatment is effective then readmission of patients reduces.
# 4. `Safety of care` - needs to be above the national average i.e., patients safety must be at most priority, avoid complications
# 5. `Mortality` - patients dying due to any complications or infections after surgery or during treatment needs to be minimised.
