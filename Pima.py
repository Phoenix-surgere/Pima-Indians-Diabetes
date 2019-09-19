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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, accuracy_score as acc
from sklearn.feature_selection import RFE
import copy
import numpy as np 

#Reproducibility seeding
np.random.seed(2)
from tensorflow import set_random_seed
set_random_seed(2)
from hyperopt import fmin, hp, tpe

#0: No diabetes, 1:Diabetic
data = pd.read_csv('diabetes.csv')
targets = data.loc[:, 'Outcome']

#1. TSNE - Visualization mostly
tsne_data = copy.deepcopy(data)
def tsne_visual(dataset):
    tsne = TSNE()
    tsne_features = tsne.fit_transform(dataset)
    dataset['tsne_2d_one'] = tsne_features[:, 0]
    dataset['tsne_2d_two'] = tsne_features[:, 1]
    sns.scatterplot(x='tsne_2d_one', y='tsne_2d_two', data=dataset, hue='Glucose') 
    dataset = dataset.iloc[:, :-3]
    return dataset
tsne_data = tsne_visual(tsne_data)
plt.show()

#2. Dimension reduction with various models: RF; LR; XGBOOST; GradB; All w/ RFE as it is more conservative
data = data.iloc[:, :-1] 

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.15)

#Instantiating all models here with some basic values
forest = RFC(n_estimators=250, random_state=42)
gbc = GBC(n_estimators=250, random_state=42)
xgbc = xgb.XGBClassifier(objective='reg:logistic', n_estimators=250, seed=42)
logit = LR(solver='lbfgs', max_iter=300 ,random_state=42)

def model_reduce(estimator, n_features, X,y, verbose=1):
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    rf_mask = rfe.support_
    if verbose == 1:
        rfe_best_features(estimator, X, rfe)
    else:
        pass
    return rf_mask

def rfe_best_features(model, data, rfe):
    '''Lower ranking= Better'''
    model_name = model.__class__.__name__
    rfe_order = pd.Series(dict(zip(data.columns, rfe.ranking_))).sort_values()
    rfe_order.rename_axis(model_name,inplace=True)
    print('\n', rfe_order)
    

logit_mask = model_reduce(logit, 4, X_train, y_train, verbose=0)
rf_mask = model_reduce(forest, 4, X_train, y_train, verbose=0)
gbc_mask = model_reduce(gbc, 4, X_train, y_train, verbose=0)
xgbc_mask = model_reduce(xgbc, 4, X_train, y_train, verbose=0)

votes = np.sum([logit_mask, rf_mask, gbc_mask, xgbc_mask], axis=0)
print(dict(zip(X_train.columns,votes)))
meta_mask = votes >= 3

data_reduced = data.loc[:, meta_mask]
print(data_reduced.columns)

#Modelling with Reduced dataset
X_train, X_test, y_train, y_test = train_test_split(data_reduced, targets, test_size=0.15)

space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
        }

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = xgb.XGBClassifier(
        objective='reg:logistic',
        n_estimators=250,
        learning_rate=0.05,
        seed=42,
        n_jobs=2,
        **params
    )
    
    best_score = cross_val_score(clf, X_train, y_train, scoring='accuracy', 
                            cv=10).mean()
    print("Acc {:.3f} params {}".format(best_score, params))
    loss = 1 - best_score
    return loss

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

#Instantiating XGBoost with best hypers found by HyperOpt
best_clf = xgb.XGBClassifier(
        objective='reg:logistic',
        n_estimators=250,
        learning_rate=0.05,
        seed=42,
        n_jobs=2,
        colsample_bytree= best['colsample_bytree'],
        gamma= best['gamma'],
        max_depth= int(best['max_depth']))

#Exploring various Classification metrics and graphs
from plot_confusion_matrix import plot_matrix
from sklearn.metrics import roc_curve,roc_auc_score, f1_score #classification_report as report,
from sklearn.metrics import average_precision_score as aps, precision_recall_curve as prc
best_clf.fit(X_train, y_train)
accuracy = acc(y_test, best_clf.predict(X_test))
confMatrix = cm(y_test, best_clf.predict(X_test))
cm_plot_labels = ['Healthy', 'Diabetic']
plot_matrix(confMatrix, cm_plot_labels, title='Confusion Matrix',normalize=True)
plt.show()

probs = best_clf.predict_proba(X_test) #default threshold is 0.5
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
f1_ration = f1_score(y_test, best_clf.predict(X_test))
avg_precision_score = aps(y_test, probs)

#ROC-AUC Curve => Best for balanced data
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

#Recall-precision curve => Best for less balanced data
precision, recall, thresholds = prc(y_test, probs)
plt.plot(recall, precision)
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.grid(True)
plt.show()
