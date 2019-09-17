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
from sklearn.linear_model import LogisticRegressionCV as LRCV, LogisticRegression as LR
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS
#import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, accuracy_score as acc, auc
from sklearn.feature_selection import RFE
from helper_funcs import rf_feature_importances
import numpy as np 
np.random.seed(2)
from tensorflow import set_random_seed
set_random_seed(2)
from hyperopt import fmin, hp, tpe

data = pd.read_csv('diabetes.csv')
targets = data.loc[:, 'Outcome']
#data = data.iloc[:, :-1]

#Enseble for feature selection: RF (features); RF+RFE; LR; XGBOOST; GradB
#from sklearn.manifold import TSNE
#1. TSNE - Visualization mostly
#tsne = TSNE()
#tsne_features = tsne.fit_transform(data)
#data['tsne_2d_one'] = tsne_features[:, 0]
#data['tsne_2d_two'] = tsne_features[:, 1]
#sns.scatterplot(x='tsne_2d_one', y='tsne_2d_two', data=data, hue='Outcome') 
#data = data.iloc[:, :-3]

#2. Dimension reduction: RFE For Trees is more conservative than feature imps
data = data.iloc[:, :-1] #COMMENT IT OUT IF RUNNING TSNE DUE TO CONFLICT
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
#print(dict(zip(X_train.columns, rfe.ranking_)))  #visual inspection, low=best

## Print the features that are not eliminated
#print(X_train.columns[rfe.support_])  #dim reduc, where the [] is the mask
#
## Calculates the test set accuracy
#accur = acc(y_test, rfe.predict(X_test))
#print("{0:.1%} accuracy on test set.".format(accur)) 
#print('*'*10)

#logitCV = LRCV(penalty='l1',solver='liblinear',cv=5,random_state=42)
#logitCV.fit(X_train, y_train)
#score = acc(y_test, logitCV.predict(X_test)); print('LogCV Acc: {}'.format(score))

#forest.fit(X_train, y_train)
## Print the importances per feature
#print(dict(zip(X_train.columns, forest.feature_importances_.round(2))))
#
#accur = acc(y_test,forest.predict(X_test))
## Print accuracy
#print("{0:.1%} accuracy on test set.".format(accur))
#
#rf_mask = forest.feature_importances_.round(2) >= 0.1
#rf_feature_importances(X_train, forest.feature_importances_)


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

#Modelling with Reduced dataset
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
    
    score = cross_val_score(clf, X_train, y_train, scoring='accuracy', 
                            cv=10).mean()
    print("Acc {:.3f} params {}".format(score, params))
    return score

space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)
