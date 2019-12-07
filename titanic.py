import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# $matplotlib inline

import plotly.offline as py
# py.init_notebook_mode(connected = True)
import chart_studio.grid_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Use these 5 base models for the stacking ?????
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

# Load in the data
train = pd.read_csv('./train.csv')
test  = pd.read_csv('./test.csv')
PassengerId = test['PassengerId']
PassengerId
list(test.columns.values)
test.head()
train.head()

full_data = [train, test]

## Feature Engineering; tease out relavant features ---------------------------

## 1. Length of passenger name
train['Name_Length'] = train['Name'].apply(len)
test['Name_Length'] = test['Name'].apply(len)

## 2. Create cabin status variable for each passenger
train['cabin_yes'] = train['Cabin'].apply(lambda x: 0 if type(x) == float
                                          else 1)
test['cabin_yes'] = test['Cabin'].apply(lambda x: 0 if type(x) == float
                                        else 1)

## 3. Create FamilySize variable from SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

## 4. Create Solo variable from FamilySize
for dataset in full_data:
    dataset['Solo'] = 0

dataset.loc[dataset['FamilySize'] == 1, 'Solo'] = 1

for dataset in full_data:  # Remove NULL values from 'Embarked' column
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

  # Remove NULL values from 'Fare' column, create CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

for dataset in full_data:  # Create CategoricalAge variable
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(low = age_avg - age_std,
                                             high = age_avg + age_std,
                                             size = age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Create Title variable from titles in passenger names:
def get_title(name): # define quick function to do the Title extraction
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:  # Now use our function to generate Title variable
    dataset['Title'] = dataset['Name'].apply(get_title)
for dataset in full_data:  # Now gather the uncommon names into "Rare" level
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt',
                                                 'Col', 'Don', 'Dr',
                                                 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'],
                                                'Rare')
    dataset['Title'] = dataset['Title'].replace('Mll', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

'''
WARNING!!!! The following lines map numeric values to NON-ordered!!
-- Go back and correct this: these should all be factors with
   ordered levels = FALSE
'''

for dataset in full_data:
    ## Mapping Sex:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    ## Mapping Title:
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    ## Mapping Embarked:
    dataset['Embarked'] = (dataset['Embarked']
                           .map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int))

    ## Mapping Fare:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) &
                (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) &
                (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    ## Mapping Age (Approriate to have ordered levels, here)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16 & dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32 & dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48 & dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    
