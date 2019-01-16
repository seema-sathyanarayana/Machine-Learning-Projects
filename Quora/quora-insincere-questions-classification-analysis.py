
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))
#os.getcwd()
#os.chdir("../input")
# Any results you write to the current directory are saved as output.


#  ### <font color="Red" size=4>" Importing all the packages for implementing the models"</font>

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords


# In[3]:


df_train= pd.read_csv("train.csv")
df_sample= pd.read_csv("sample_submission.csv")
df_test= pd.read_csv("test.csv")

print("The no of rows in train datasets is {} and columns is {}".format(df_train.shape[0],df_train.shape[1]))
print("The no of rows in test datasets is {} and columns is {}".format(df_test.shape[0],df_test.shape[1]))


# ## <font color="Red"> Exploratory Data Analysis of Datasets</font>

# * ### <font color="orange">Checking if there is missing value in the datasets</font>

# In[4]:


df_train.isnull().sum() # There is no missing value in the train datasets.Good to go!!


# ### Description of the datasets types

# In[5]:


df_train.info()


# In[6]:


df_test.info()


# ### <font color="orange">Visualization of target datasets</font>

# In[7]:


plt.xlabel("Target")
plt.ylabel("Total Count")
df_train['target'].hist()


# ## <font color="Red">Implementing the models to predict the accuracy</font>

#  ### <font color="Green">1. Decision Tree</font>

# In[8]:


import string
import nltk
nltk.download('stopwords')
eng_stopwords = set(stopwords.words("english")) #used to extract stop words like 'is', 'there', etc from lib nltk

from sklearn import tree
from sklearn.metrics import accuracy_score

#eng_stopwords


# ### <font color="orange">Feature Columns Setting</font>

# ### Let Extract metadata from feature columns to gain meaningful insights

# In[9]:


## Number of words in the text ##
df_train["num_words"] = df_train["question_text"].apply(lambda x: len(str(x).split()))
df_test["num_words"] = df_test["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
df_train["num_unique_words"] = df_train["question_text"].apply(lambda x: len(set(str(x).split())))
df_test["num_unique_words"] = df_test["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
df_train["num_chars"] = df_train["question_text"].apply(lambda x: len(str(x)))
df_test["num_chars"] = df_test["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
df_train["num_stopwords"] = df_train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
df_test["num_stopwords"] = df_test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
df_train["num_punctuations"] =df_train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
df_test["num_punctuations"] =df_test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
df_train["num_words_upper"] = df_train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
df_test["num_words_upper"] = df_test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
df_train["num_words_title"] = df_train["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df_test["num_words_title"] = df_test["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
df_train["mean_word_len"] = df_train["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df_test["mean_word_len"] = df_test["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[10]:


df_train.head(2)


# In[11]:


df_test.head(2)


# In[12]:


df_train['num_words'].loc[df_train['num_words']>80] = 80 #truncation for better visuals
plt.figure(figsize=(12,8))
sns.stripplot(y='num_words', data=df_train)
plt.title("Number of words in question text", fontsize=15)
plt.show()


# In[13]:


df_test


# In[14]:


#y_train= df_train['qid'].values
#y_test=  df_test['qid'].values
y_train= df_train['target'].values
X_train=df_train.drop(['qid','question_text','target'],axis=1)
X_test=df_test.drop(['qid','question_text'],axis=1)


# ### Training the Model

# In[15]:


from sklearn import tree
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
my_tree_one


# In[16]:


my_tree_one.fit(X_train,y_train)


# ### feature Importance value

# In[17]:


list(zip(X_train.columns,my_tree_one.feature_importances_))


# In[18]:


# The accuracy of the model
print(my_tree_one.score(X_train, y_train))


# In[19]:


# Predictions from Decision Tree Model
y_pred = my_tree_one.predict(X_test)


#  ### <font color="Green">2. Random Forest</font>

# In[20]:


# Building and fitting Random Forest
from sklearn.ensemble import RandomForestClassifier

forest_one = RandomForestClassifier(max_depth = 10, n_estimators = 100, random_state = 1)

# Fitting the model on Train Data
my_forest = forest_one.fit(X_train, y_train)


# In[21]:


# Print the accuracy score of the fitted random forest
print(my_forest.score(X_train, y_train))


# In[22]:


# Making predictions of test data
pred_for = my_forest.predict(X_test)


# In[23]:


list(zip(X_train.columns,my_forest.feature_importances_))


#  ### <font color="Green">3. Grid Search</font>

# In[24]:


# Different parameters we want to test

max_depth = [3,5,7] 
criterion = ['gini', 'entropy']


# In[25]:


# Importing GridSearch
from sklearn.grid_search import GridSearchCV

# Building the model
my_tree_two = tree.DecisionTreeClassifier()
grid = GridSearchCV(estimator = my_tree_two, cv=3, 
                    param_grid = dict(max_depth = max_depth, criterion = criterion))


# In[26]:


grid.fit(X_train,y_train)


# In[27]:


# Best accuracy score
grid.best_score_


# In[28]:


# Best params for the model
grid.best_params_


# In[29]:


# Building the model based on new parameters
my_tree_two = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 7, random_state=42)
my_tree_two.fit(X_train,y_train)


# In[30]:


# Accuracy Score for new model
my_tree_two.score(X_train,y_train)


# ### <font color="Green">5. Naive Bayes</font>

# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[32]:


stopset=set(stopwords.words("english"))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# In[33]:


x_fit=vectorizer.fit_transform(X_train)


# In[34]:


mnb=MultinomialNB()
model=mnb.fit(X_train,y_train)


# In[35]:


y_prd=model.predict(X_test)


# In[41]:


model.score(X_train,y_train)


# ### Conclusion : All the above algorithms have accuracy score 94%
