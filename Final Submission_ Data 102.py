#!/usr/bin/env python
# coding: utf-8

# ## Research Question 1: Causal Effect between Party Support and candidates being on the ballot.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import interact, interactive
import itertools
import hashlib
from scipy.stats import poisson, norm, gamma
import statsmodels.api as sm
  
sns.set(style="dark")
plt.style.use("ggplot")

from pymc3 import *
import pymc3 as pm

import arviz as az


# In[2]:


dem = pd.read_csv("dem_candidates.csv")


# In[3]:


dem = dem.dropna(subset=['Party Support?'])
print(dem.shape)
dem.head()


# In[4]:


supported = dem[dem['Party Support?'] == 'Yes']
supported[supported['Primary Status'] == 'Advanced'].head()


# ### EDA

# Visualization of the number of cadidates being on the ballot and not being on the ballot given their party support status.

# In[5]:


dem_short = dem[['Party Support?', 'General Status']]


# In[6]:


support_and_status = dem_short.groupby(by=['Party Support?', 'General Status'], as_index=False).size()
support_and_status = support_and_status.rename(columns={'size': 'Number of candidates'})
support_and_status


# In[7]:


sns.barplot(x='Party Support?', y='Number of candidates', hue='General Status', data=support_and_status)
plt.ylabel('Number of candidates');


# In[8]:


support_and_status['Percentage of candidates'] = support_and_status['Number of candidates'] / support_and_status.groupby('Party Support?')['Number of candidates'].transform('sum') * 100


# In[9]:


support_and_status


# In[10]:


sns.barplot(x='Party Support?', y='Percentage of candidates', hue='General Status', data=support_and_status)
plt.ylabel('Percentage of candidates %');


# From the bar chart above, we can see that candidates with party support have higher chance being on the ballot, and candidates without party support have higher chance not being on the ballot. Thus, there might be some correlation between `Party Support?` and `General Status`. This visualization is relevant to our research question because it suggests that there might be a causation between `Party Support?` and `General Status` of a candidate.

# ### Research Question Results

# In[11]:


dem['General Status'] = dem['General Status'].map({'None': 0, 'On the Ballot': 1})
dem['Veteran?'] = dem['Veteran?'].map({'No': 0, 'Yes': 1})
dem['LGBTQ?'] = dem['LGBTQ?'].map({'No': 0, 'Yes': 1})
dem['Elected Official?'] = dem['Elected Official?'].map({'No': 0, 'Yes': 1})
dem['STEM?'] = dem['STEM?'].map({'No': 0, 'Yes': 1})
dem['Obama Alum?'] = dem['Obama Alum?'].map({'No': 0, 'Yes': 1})
dem['Self-Funder?'] = dem['Self-Funder?'].map({'No': 0, 'Yes': 1})
dem['Race'] = dem['Race'].map({'Nonwhite': 0, 'White': 1})
dem['Won Primary'] = dem['Won Primary'].map({'No': 0, 'Yes': 1})
dem['Party Support?'] = dem['Party Support?'].map({'No': 0, 'Yes': 1})
dem['Biden Endorsed?'] = dem['Biden Endorsed?'].map({'No': 0, 'Yes': 1})
dem['Guns Sense Candidate?'] = dem['Guns Sense Candidate?'].map({'No': 0, 'Yes': 1})


# #### Causal Inference:

# ##### Methods:
# 
# • Treatment: `Party Support?`
# 
# • Outcome: `General Status`
# 
# • (Possible) Confounders: `Race`, `LGBTQ?`, `STEM?`,`Obama Alum?`
# 
# (other possible confounders are omitted due to the lack of data.)
# 
# We use outcome regression to conduct our research. There is no colliders given the `General Status` is the final result.

# In[12]:


def fit_LR_model(df, target_variable, explanatory_variables, intercept = False):
    """
    Fits an Logistic Regression model from data.
    
    Inputs:
        df: pandas DataFrame
        target_variable: string, name of the target variable
        explanatory_variables: list of strings, names of the explanatory variables
        intercept: bool, if True add intercept term
    Outputs:
        fitted_model: model containing OLS regression results
    """
    
    target = df[target_variable]
    inputs = df[explanatory_variables]
    if intercept:
        inputs = sm.add_constant(inputs)
    
    fitted_model = sm.Logit(target, inputs).fit()
    return(fitted_model)


# In[13]:


dem_clean = dem.dropna(subset=['General Status', 'Party Support?', 'Race', 'LGBTQ?', 'STEM?','Obama Alum?'])
print(dem_clean.shape)


# In[14]:


linear_model = fit_LR_model(dem_clean, 'General Status', ['Party Support?'])
print(linear_model.summary())


# ##### Our models:

# In[15]:


linear_model = fit_LR_model(dem_clean, 'General Status', ['Party Support?', 'Race', 'LGBTQ?', 'STEM?','Obama Alum?'])
print(linear_model.summary())


# ##### Results:
# 
# From the model above, being supported by the party increases the log odds ratio of the candidate being on the ballot by between 4.1 to 9.8. The causation is solidified by significantly small p-value. If the model is misspecified, then the interpretation is only valid with respect to the projected model.
# 
# For all factors other than `Party Support?`, the p-values are significant for `Race` and `STEM`.

# Our model's Log-Likelihood is also closer to 0 than the model not considering confounders. Thus, it is more accurate.

# ## Research Question 2: Predicting Democratic Primary Election Results From Personal Information on Candidates using Logistic Regression vs. Decision Tree Classifier

# In[16]:


dem = pd.read_csv("dem_candidates.csv")
dems = dem[['Candidate', 'State', 'Veteran?', 'LGBTQ?', 'Elected Official?', 'STEM?', 'Obama Alum?', 'Self-Funder?']]
dems['win'] = dem['Primary Status'].map({'Lost': 0, 'Advanced': 1})
dems['Veteran?'] = dem['Veteran?'].map({'No': 0, 'Yes': 1})
dems['LGBTQ?'] = dem['LGBTQ?'].map({'No': 0, 'Yes': 1})
dems['Elected Official?'] = dem['Elected Official?'].map({'No': 0, 'Yes': 1})
dems['STEM?'] = dem['STEM?'].map({'No': 0, 'Yes': 1})
dems['Obama Alum?'] = dem['Obama Alum?'].map({'No': 0, 'Yes': 1})
dems['Self-Funder?'] = dem['Self-Funder?'].map({'No': 0, 'Yes': 1})


# In[17]:


dems.head(1)


# ### EDA

# In[18]:


xnoise, ynoise = np.random.normal(0, 0.05, size = 811), np.random.normal(0, 0.05, size = 811)
plt.scatter(x = dems['Veteran?'] + xnoise, y = dems['win'] + ynoise, alpha = 0.3)
plt.xlabel("Veteran?")
plt.ylabel("Advanced?")
f = np.mean(dems[dems['Veteran?'] == 0]['win'])
t = np.mean(dems[dems['Veteran?'] == 1]['win'])
f, t;


# On average, being a veteran marginally reduces chances of advancing without controling for confounders, this might be worth a further analysis.

# In[19]:


xnoise, ynoise = np.random.normal(0, 0.05, size = 811), np.random.normal(0, 0.05, size = 811)
plt.scatter(x = dems['LGBTQ?'] + xnoise, y = dems['win'] + ynoise, alpha = 0.3)
plt.xlabel("LGBTQ?")
plt.ylabel("Advanced?")
f = np.mean(dems[dems['LGBTQ?'] == 0]['win'])
t = np.mean(dems[dems['LGBTQ?'] == 1]['win'])
f, t;


# On average, being an LGBTQ member marginally reduces chances of advancing without controling for confounders, this might be worth further analyzing.

# In[20]:


xnoise, ynoise = np.random.normal(0, 0.05, size = 811), np.random.normal(0, 0.05, size = 811)
plt.scatter(x = dems['Elected Official?'] + xnoise, y = dems['win'] + ynoise, alpha = 0.3)
plt.xlabel("Elected official?")
plt.ylabel("Advanced?")
f = np.mean(dems[dems['Elected Official?'] == 0]['win'])
t = np.mean(dems[dems['Elected Official?'] == 1]['win'])
f, t;


# On average being an Elected official increases the chances of advancement without controling for confounders, this is worth a further exploration in phase 2.

# In[21]:


xnoise, ynoise = np.random.normal(0, 0.05, size = 811), np.random.normal(0, 0.05, size = 811)
plt.scatter(x = dems['STEM?'] + xnoise, y = dems['win'] + ynoise, alpha = 0.3)
plt.xlabel("STEM?")
plt.ylabel("Advanced?")
f = np.mean(dems[dems['STEM?'] == 0]['win'])
t = np.mean(dems[dems['STEM?'] == 1]['win'])
f, t;


# On average, having a STEM background significantly reduces the chances of advancing without controling for confounders, this is surely worth exploring in phase 2. Maybe non-STEM degrees builds skills required for politicians.

# In[22]:


xnoise, ynoise = np.random.normal(0, 0.05, size = 811), np.random.normal(0, 0.05, size = 811)
plt.scatter(x = dems['Obama Alum?'] + xnoise, y = dems['win'] + ynoise, alpha = 0.3)
plt.xlabel("Obama Alum?")
plt.ylabel("Advanced?")
f = np.mean(dems[dems['Obama Alum?'] == 0]['win'])
t = np.mean(dems[dems['Obama Alum?'] == 1]['win'])
f, t;


# On average, being an Obama Alum significantly increases the chances of advancing without controling for confounders, this is surely worth exploring in phase 2. Maybe Obama alum have skills required for successful advancement.

# ### Methods

# We will be predicting whether a Democratic candidate wins the Democratic primaries or not using demographic/personal information about the candidate. The features we chose are "Obama Alum?", "Veteran?", "STEM?", "LGBTQ?", "Elected Official?". We will compare the performance of logistic regression vs decision tree classification when predicting primary election outcomes for Democratic candidates from personal information.

# In[23]:


#clean the dataset utilizing only variables of value. 
df = dems[['Candidate', 'Veteran?', 'LGBTQ?', 'Elected Official?', 'STEM?', 'Obama Alum?', 'win']].dropna()
df.head(1)


# In[24]:


#Split the data set into train and test sets
from sklearn.model_selection import train_test_split
data_tr, data_te = train_test_split(df, test_size=0.10, random_state=42)
print("Training Data Size: ", data_tr.shape)
print("Test Data Size: ", len(data_te))

# X, Y are training data
X = data_tr[['Veteran?', 'LGBTQ?', 'Elected Official?', 'STEM?', 'Obama Alum?']]
Y = data_tr['win']


# In[25]:


X.head(1)


# ### Logistic Regression

# In[26]:


# Fit Logistic GLM model on training set
freq_model = sm.GLM(Y, sm.add_constant(X), family=sm.families.Binomial())
freq_res = freq_model.fit()
print(freq_res.summary())


# In[27]:


# Fit Logistic GLM model on training set excluding LGBTQ data since it's parameter is centered around 0.
X = data_tr[['Veteran?', 'Elected Official?', 'STEM?', 'Obama Alum?']]
freq_model = sm.GLM(Y, sm.add_constant(X), family=sm.families.Binomial())
freq_res = freq_model.fit()
print(freq_res.summary())


# In[28]:


# use parameters to predict Y values for test set
X_test = data_te[['Veteran?', 'Elected Official?', 'STEM?', 'Obama Alum?']]
Y_prob = 1 / (1 + (1/np.exp(X_test['Veteran?']*(-0.2156) + X_test['Elected Official?']*0.6369 + X_test['STEM?']*(-0.3501)
                   + X_test['Obama Alum?']*0.1124 - 0.7382)))


# ### Logistic Regression Results

# In[29]:


#accuracy of training dataset
Y_prob_train = 1 / (1 + (1/np.exp(X['Veteran?']*(-0.2156) + X['Elected Official?']*0.6369 + X['STEM?']*(-0.3501)
                   + X['Obama Alum?']*0.1124 - 0.7382)))
Y_preb_train = (Y_prob_train>= 0.5)
a = np.mean(data_tr["win"] == Y_preb_train)
print(f"Accuracy on training data set for parametric case: {a}")


# In[30]:


#accuracy of parametric model on test set at 0.5 threshold
Y_test = data_te['win']
Y_pred = (Y_prob >= 0.5)
accuracy = np.mean(Y_test == Y_pred)
print(f"Accuracy on test set for parametric case: {accuracy}")


# In[31]:


#calculating logistic regression cross entropy loss
ce_loss = -(np.average((Y_test * np.log(Y_prob)) + (1 - Y_test) * np.log(Y_prob)))
ce_loss


# ### Decision Tree Classification

# In[32]:


#non parametric classifier - decision tree

X = data_tr[['Veteran?', 'LGBTQ?', 'Elected Official?', 'STEM?', 'Obama Alum?']]
X_test = data_te[['Veteran?', 'LGBTQ?', 'Elected Official?', 'STEM?', 'Obama Alum?']]

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()
model_tree.fit(X, Y)
probs = model_tree.predict_proba(X_test)[:, 1]
y_hat_tree = (probs > 0.5).astype(np.int64)


# ### Decision Tree Classification Results

# In[33]:


accuracy = np.mean(Y_test == y_hat_tree)
print(f"Accuracy on test set for non-parametric case: {accuracy}")


# In[34]:


probs_null = model_tree.predict_proba(X)[:, 1]
y_null = (probs_null > 0.5).astype(np.int64)

accuracy = np.mean(Y == y_null)
print(f"Accuracy on train set for non-parametric case: {accuracy}")

#explanation for not 100% accuracy - since we have binary RV, we can only split once on 
#each feature within each vertical path. 


# In[35]:


#interpretability and explanability of why model makes choices, best attempt below. 
#It appears classifier splitting down the middle for each train feature and
#rounding up the average of outcome variables to train itself, applying the same rule to test set. 

from sklearn.tree import plot_tree

plt.figure(figsize=(16, 12))
plot_tree(model_tree, fontsize=12, filled=True);


# We can conclude that the logistic regression model and the decision tree classification perform similarly. 

# In[ ]:




