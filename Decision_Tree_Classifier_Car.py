#!/usr/bin/env python
# coding: utf-8

# ## DECISION TREE CLASSIFIER USING Scikit-learn

# ### Creating and Visualizing a Decision Tree Classification Model in Machine Learning

# ### Problem statement:  To predict the safety of the car. 
# 
# In this project, I have build a Decision Tree Classifier to predict the safety of the car. I have implemented Decision Tree Classification with Python and Scikit-Learn.

# ### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings

warnings.filterwarnings('ignore')


#  ### Import dataset

# In[2]:


data = '/Users/sushmapawar/Desktop/Projects/Decision_Tree_dataset/car.data'

df = pd.read_csv(data, header=None)
df


# ###  Exploratory data analysis

# In[3]:


# Dimensions of dataset

df.shape


# In the dataset there are 1728 instances(rows) and 7 variables(columns).

# In[4]:


# View the dataset

df.head()


# #### Rename column names
# The columns are merely labelled as 0,1,2.... and so on. We should give proper names to the columns.

# In[5]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df.columns = col_names

df.columns


# In[6]:


# Review the dataset

df.head()


# Column names are renamed.Columns have meaningful names.

# ### View summary of dataset

# In[7]:


df.info()


# ### Frequency distribution of values in variables
# Check the frequency counts of categorical variables.

# In[8]:


col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:   
    print(df[col].value_counts())   


# We can see that the doors and persons are categorical in nature. So,treat them as categorical variables.

# ### Summary of variables
# There are 7 variables(columns) in the dataset. All the variables are of categorical data type.
# 
# These are given by buying, maint, doors, persons, lug_boot, safety and class.
# 
# class is the target variable.

# #### Counts of class variable

# In[9]:


df['class'].value_counts()


# The class target variable is ordinal in nature.

# In[10]:


df['class'].value_counts().plot(kind = 'bar')
plt.show()


# In[11]:


df['safety'].value_counts().plot(kind = 'bar')
plt.show()


# #### Missing(NULL) values in variables

# In[12]:


# check missing values in variables

df.isnull().sum()


# There are no missing values in the dataset

#  ### Declare feature  and target variable

# In[13]:


X = df.drop(['class'], axis=1)
y = df['class']


# ### Split data into separate training and test set

# In[14]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[15]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# ### Feature Engineering
# Feature Engineering is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power.
# 
# Check the data types of variables again.

# In[16]:


# check data types in X_train
X_train.dtypes


# #### Encode categorical variables
# Encode the categorical variables.

# In[17]:


X_train.head()


# All the variables are ordinal categorical data type.

# In[18]:


get_ipython().system('pip install category_encoders')


# In[19]:


# import category encoders

import category_encoders as ce


# In[20]:


# encode variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[21]:


X_train.head()


# In[22]:


X_test.head()


# Training and Test set ready for model building.

#  ### Decision Tree Classifier with criterion gini index

# In[23]:


# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier


# In[24]:


# instantiate the DecisionTreeClassifier model with criterion gini index

dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)


# fit the model
dt_gini.fit(X_train, y_train)


# In[25]:


dt_gini.get_params()


# ### Predict the Test set results with criterion gini index

# In[26]:


y_pred_gini = dt_gini.predict(X_test)


# ### Check accuracy score with criterion gini index

# In[27]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))


# ### Check precision score with criterion gini index

# In[28]:


from sklearn.metrics import precision_score
print('Model precision score with criterion gini index: {0:0.4f}'. format(precision_score(y_test, y_pred_gini,average='weighted')))


# ### Check recall score with criterion gini index

# In[29]:


from sklearn.metrics import recall_score
print('Model recall score with criterion gini index: {0:0.4f}'. format(recall_score(y_test, y_pred_gini,average='weighted')))


# y_test are the true class labels and y_pred_gini are the predicted class labels in the test-set.

# In[30]:


y_pred_train_gini = dt_gini.predict(X_train)

y_pred_train_gini


# In[31]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))


# ### Compare the train-set and test-set accuracy
# Compare the train-set and test-set accuracy to check for overfitting.
# #### Check for overfitting and underfitting

# In[32]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(dt_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(dt_gini.score(X_test, y_test)))


# The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.

# In[33]:


feature_names = X.columns
feature_names


# In[34]:


dt_gini.feature_importances_


# In[35]:


feature_importance = pd.DataFrame(dt_gini.feature_importances_,index = feature_names).sort_values(0,ascending= False)
feature_importance


# In[36]:


features = list(feature_importance[feature_importance[0]>0].index)
features


# In[37]:


feature_importance.head(10).plot(kind='bar')


# In[38]:


from sklearn import tree
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt_gini,feature_names=feature_names,class_names=None
                   , filled=True,fontsize=25)


# ### Decision Tree Classifier with criterion entropy

# In[39]:


# instantiate the DecisionTreeClassifier model with criterion entropy
dt_en = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
# fit the model
dt_en.fit(X_train, y_train)


# ### Predict the Test set results with criterion entropy

# In[40]:


y_pred_en = dt_en.predict(X_test)


# ### Check accuracy score with criterion entropy

# In[41]:


from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))


# ### Check precision score with criterion entropy

# In[42]:


from sklearn.metrics import precision_score
print('Model precision score with criterion entropy: {0:0.4f}'.format(precision_score(y_test, y_pred_en,average='weighted')))


# ### Check recall score with criterion entropy

# In[43]:


from sklearn.metrics import recall_score
print('Model recall score with criterion entropy: {0:0.4f}'.format(recall_score(y_test, y_pred_en,average='weighted')))


# In[44]:


y_pred_train_en = dt_en.predict(X_train)

y_pred_train_en


# In[45]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


# ### Compare the train-set and test-set accuracy
# Compare the train-set and test-set accuracy to check for overfitting.

# ### Check for overfitting and underfitting

# In[46]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(dt_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(dt_en.score(X_test, y_test)))


# The training-set score and test-set score is same as above. The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.
# 
# Now, based on the above analysis we can conclude that our classification model accuracy is very good.
# Our model is doing a very good job in terms of predicting the class labels.
# 
# But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.

# In[47]:


dt_en.feature_importances_


# In[48]:


feature_importance = pd.DataFrame(dt_en.feature_importances_,index = feature_names).sort_values(0,ascending= False)
feature_importance


# In[49]:


features = list(feature_importance[feature_importance[0]>0].index)
features


# In[50]:


feature_importance.head(10).plot(kind='bar')


# In[51]:


from sklearn import tree
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt_en,feature_names=feature_names,class_names=None
                   , filled=True,fontsize=25)


# ### Confusion matrix
# A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.
# 
# Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-
# 
# True Positives (TP) – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
# 
# True Negatives (TN) – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
# 
# False Positives (FP) – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.
# 
# False Negatives (FN) – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.
# 
# These four outcomes are summarized in a confusion matrix given below.

# In[52]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

entropy_c = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', entropy_c)


# In[53]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

gini_c = confusion_matrix(y_test, y_pred_gini)

print('Confusion matrix\n\n', gini_c)


# #### Classification Report
# Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. I have described these terms in later.
# 
# We can print a classification report as follows:-

# In[54]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))


# In[55]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_gini))


# ### Results and conclusion
# 1. In this project, I have build a Decision-Tree Classifier model to predict the safety of the car. I build two models, one with criterion gini index and another one with criterion entropy. The model yields a very good performance as indicated by the model accuracy in both the cases which was found to be 0.8021.
# 2. In the model with criterion gini index, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.
# 3. Similarly, in the model with criterion entropy, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021.We get the same values as in the case with criterion gini. So, there is no sign of overfitting.
# 4. In both the cases, the training-set and test-set accuracy score is the same. It may happen because of small dataset.
# 5. The confusion matrix and classification report yields very good model performance.

# In[ ]:




