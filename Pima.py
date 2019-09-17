# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:36:35 2019

@author: black
"""

#EASIER first dataset (less features-no categorical) for DimRed &Classification
import pandas as pd
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, accuracy_score as acc, auc
from sklearn.feature_selection import RFE
import numpy as np 
import seaborn as sns

#Setting Seeds for reproducibility reasons
np.random.seed(2)
from tensorflow import set_random_seed
set_random_seed(2)
from hyperopt import fmin, hp, tpe

data = pd.read_csv('diabetes.csv')
targets = data.loc[:, 'Outcome']

#Enseble for feature selection (via RFE): RF; LR; XGBOOST; GradB
#1. TSNE - Visualization mostly
tsne = TSNE()
tsne_features = tsne.fit_transform(data)
data['tsne_2d_one'] = tsne_features[:, 0]
data['tsne_2d_two'] = tsne_features[:, 1]
sns.scatterplot(x='tsne_2d_one', y='tsne_2d_two', data=data, hue='Outcome') 
data = data.iloc[:, :-3]

#2. Dimension reduction: RFE For Trees is more conservative than feature importances
#data = data.iloc[:, :-1] #COMMENT IT OUT IF RUNNING TSNE DUE TO CONFLICT, else use that
ss = SS()

#data[data.columns] = ss.fit_transform(data[data.columns]) # 1st try w/out scale
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.15)

forest = RFC(n_estimators=250, random_state=42)
gbc = GBC(n_estimators=250, random_state=42)
xgbc = xgb.XGBClassifier(objective='reg:logistic', n_estimators=250, seed=42)
logit = LR(solver='lbfgs', max_iter=300 ,random_state=42)

rfe = RFE(estimator=logit, n_features_to_select=4, verbose=1)
rfe.fit(X_train, y_train)
logit_mask = rfe.support_

rfe = RFE(estimator=forest, n_features_to_select=4, verbose=1)
rfe.fit(X_train, y_train)
rf_mask = rfe.support_


rfe = RFE(estimator=gbc, n_features_to_select=4, verbose=1)
rfe.fit(X_train, y_train)
gbc_mask = rfe.support_


rfe = RFE(estimator=xgbc, n_features_to_select=4, verbose=1)
rfe.fit(X_train, y_train)
xgbc_mask = rfe.support_

votes = np.sum([logit_mask, rf_mask, gbc_mask, xgbc_mask], axis=0)
print(dict(zip(X_train.columns,votes)))
meta_mask = votes >= 3

data_reduced = data.loc[:, meta_mask]
print(data_reduced.columns)

#Modelling with Reduced dataset with XGB and basic Hyperopt tuning
X_train, X_test, y_train, y_test = train_test_split(data_reduced, targets, test_size=0.15)


def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        n_jobs=4,
        **params
    )
    
    best_score = cross_val_score(clf, X_train, y_train, scoring='accuracy', 
                            cv=10).mean()
    print("Acc {:.3f} params {}".format(best_score, params))
    #The score function should return the loss (1-score)
    # since the optimize function looks for the minimum - Hyperopt peculiarity, need to remember that!
    loss = 1 - best_score
    return loss
space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)
