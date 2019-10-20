import pandas as pd
# for upload
import base64

# Model Imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import os

df = pd.read_csv('myproject/dashboard/student_data.csv')




# Model
# ------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Preprocessing

df['total score'] = df['math score'] + df['reading score'] + df['writing score']

average = df['total score'].mean()
# No one scored exacly the mean, so we can leave the 'equal' operation out.
df['performance'] = np.where(df['total score'] > average, 'above average', 'below average')
df[['total score', 'performance']].head()

# One Hot Encoding predictors

df_encoded = pd.get_dummies(df, columns=['gender',
                                         'parental level of education',
                                         'lunch', 'test preparation course'], drop_first=True)

# Train Test Split

X = df_encoded.drop(['math score', 'reading score', 'writing score', 'total score',
                     'performance', 'race/ethnicity'], axis=1).values  # add values to transform df to np array
y = df['performance'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# RGB SVM + GSCV for Hyperparameter Tuning

# RBF SVM
svm = SVC(kernel='rbf')

# GridSearchCV
parameters = {'C': [0.1, 1, 10], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
clf = GridSearchCV(svm, parameters, cv=5)
clf.fit(X_train, y_train)

# Dashboard
# ------------------------------------------------------------------------------------------------------------------------------------------------------- #

subjects = ['math score', 'reading score', 'writing score']
