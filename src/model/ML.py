#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2019, Jianfeng Chen <jchen37@ncsu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, _distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

import os
import sys
import random
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]
sys.path.append(f'{root}/src')

from model.utilities import get_score
from model.hp import Hyperparameter


def _define_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    _, indices = nbrs.kneighbors(data)
    for _ in range(0, num):
        mid = random.randint(0, len(data) - 1)
        nn = indices[mid, random.randint(1, k - 1)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random.random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    corpus = np.vstack((corpus, np.array(data)))
    return corpus


def _apply_model(train_df, test_df, model):
    X = train_df.loc[:, train_df.columns != 'label']
    y = train_df.label
    model.fit(X, y)
    X_test = test_df.loc[:, test_df.columns != 'label']
    prediction = model.predict(X_test)
    return prediction


def SMOTE(train_df, HP):
    m, k, r = HP['SMOTE']['m'], HP['SMOTE']['k'], HP['SMOTE']['r']
    X = train_df.loc[:, train_df.columns != 'label']
    y = train_df.label
    pos_train, neg_train = X.loc[y == 1], X.loc[y == 0]

    pos_train = _define_smote(pos_train.values, m, k, r)
    neg_train = neg_train.sample(min(m, neg_train.shape[0])).values
    X = np.concatenate((pos_train, neg_train), axis=0)
    y = [1] * pos_train.shape[0] + [0] * neg_train.shape[0]
    y = np.asarray(y).reshape(-1, 1)
    balanced = pd.DataFrame(
        np.concatenate((X, y), axis=1), columns=train_df.columns).astype('int')
    return balanced


def DT(train_df, test_df, HP):
    # a = _randuniform(0.0, 1.0)
    # b = _randchoice(['gini', 'entropy'])
    # c = _randchoice(['best', 'random'])
    a, b, c = HP['DT']['min_samples_split'], HP['DT']['criterion'], HP['DT'][
        'splitter']
    model = DecisionTreeClassifier(
        criterion=b,
        splitter=c,
        min_samples_split=a,
        max_features=None,
        min_impurity_decrease=0.0)
    return _apply_model(train_df, test_df, model)


def RF(train_df, test_df, HP):
    # a = _randint(50, 150)
    # b = _randchoice(['gini', 'entropy'])
    # c = _randuniform(0.0, 1.0)
    a, b, c = HP['RF']['n_estimators'], HP['RF']['criterion'], HP['RF'][
        'min_samples_split']
    model = RandomForestClassifier(
        n_estimators=a,
        criterion=b,
        min_samples_split=c,
        max_features=HP['RF']['max_features'],
        max_leaf_nodes=HP['RF']['max_leaf_nodes'],
        max_depth=HP['RF']['max_depth'],
        min_impurity_decrease=0.0,
        n_jobs=-1)
    return _apply_model(train_df, test_df, model)


def SVM(train_df, test_df, HP):
    # a = _randint(1, 500)
    # b = _randchoice(['linear', 'poly', 'rbf', 'sigmoid'])
    # c = _randint(2, 10)
    # d = _randuniform(0.0, 1.0)
    # e = _randuniform(0.0, 0.1)
    # f = _randuniform(0.0, 0.1)
    a, b, c, d, e, f = HP['SVM']['C'], HP['SVM']['kernel'], HP['SVM'][
        'degree'], HP['SVM']['gamma'], HP['SVM']['coef0'], HP['SVM']['tol']
    model = SVC(
        C=float(a),
        kernel=b,
        degree=c,
        gamma=d,
        coef0=e,
        tol=f,
        cache_size=20000)
    return _apply_model(train_df, test_df, model)


def KNN(train_df, test_df, HP):
    # a = _randint(2, 25)
    # b = _randchoice(['uniform', 'distance'])
    # c = _randchoice(['minkowski', 'chebyshev'])
    a, b, c, d = HP['KNN']['n_neighbors'], HP['KNN']['weights'], HP['KNN'][
        'metric'], HP['KNN']['p']
    if c != 'minkowski':
        d = 2
    model = KNeighborsClassifier(
        n_neighbors=a, weights=b, algorithm='auto', p=d, metric=c, n_jobs=-1)
    return _apply_model(train_df, test_df, model)


def NB(train_df, test_df, HP):
    model = GaussianNB(priors=None, var_smoothing=HP['NB']['var_smoothing'])
    return _apply_model(train_df, test_df, model)


def LR(train_df, test_df, HP):
    # a = _randchoice(['l1', 'l2'])
    # b = _randuniform(0.0, 0.1)
    # c = _randint(1, 500)
    a, b, c = HP['LR']['penalty'], HP['LR']['tol'], HP['LR']['C']
    model = LogisticRegression(
        penalty=a,
        tol=b,
        C=float(c),
        solver='liblinear',
        multi_class='warn',
        n_jobs=-1)
    return _apply_model(train_df, test_df, model)


def evaluation(dataset, HP):
    # read csc file (MacBook)
    train_df = pd.read_csv(f'{root}/data/FARSEC/{dataset}-train.csv').drop(
        ['id'], axis=1)
    test_df = pd.read_csv(f'{root}/data/FARSEC/{dataset}-test.csv').drop(
        ['id'], axis=1)
    train_df = SMOTE(train_df, HP)

    if 'DT' in HP.keys():
        prediction = DT(train_df, test_df, HP)
    elif 'RF' in HP.keys():
        prediction = RF(train_df, test_df, HP)
    elif 'SVM' in HP.keys():
        prediction = SVM(train_df, test_df, HP)
    elif 'KNN' in HP.keys():
        prediction = KNN(train_df, test_df, HP)
    elif 'LR' in HP.keys():
        prediction = LR(train_df, test_df, HP)
    elif 'NB' in HP.keys():
        prediction = NB(train_df, test_df, HP)
    else:
        assert False, "check here"

    test_labels = test_df.label.values.tolist()
    rec = get_score('recall', prediction, test_labels, "NA")
    fpr = get_score('false_alarm', prediction, test_labels, "NA")
    gm = get_score("g_measure", prediction, test_labels, "NA")

    return np.array([rec, 1 - fpr, gm])


def get_HP_obj():
    FARSEC_HP = Hyperparameter(
        preprocessing_names=['SMOTE'],
        learner_names=['DT', 'RF', 'SVM', 'KNN', 'NB', 'LR'])

    FARSEC_HP.register_hp('SMOTE', 'k', np.arange(2, 20))
    FARSEC_HP.register_hp('SMOTE', 'm', np.arange(50, 400))
    FARSEC_HP.register_hp('SMOTE', 'r', np.arange(1, 6))

    FARSEC_HP.register_hp('DT', 'min_samples_split', np.arange(2, 20, 1))
    FARSEC_HP.register_hp('DT', 'criterion', ['gini', 'entropy'])
    FARSEC_HP.register_hp('DT', 'splitter', ['best', 'random'])

    FARSEC_HP.register_hp('RF', 'n_estimators', np.arange(10, 150))
    FARSEC_HP.register_hp('RF', 'criterion', ['gini', 'entropy'])
    FARSEC_HP.register_hp('RF', 'min_samples_leaf', np.arange(1, 20))
    FARSEC_HP.register_hp('RF', 'min_samples_split', np.arange(2, 20))
    FARSEC_HP.register_hp('RF', 'max_leaf_nodes', np.arange(2, 50))
    FARSEC_HP.register_hp('RF', 'max_features', np.arange(0.01, 1, 0.01))
    FARSEC_HP.register_hp('RF', 'max_depth', np.arange(1, 10))

    FARSEC_HP.register_hp('SVM', 'C', np.arange(1, 500))
    FARSEC_HP.register_hp('SVM', 'kernel',
                          ['linear', 'poly', 'rbf', 'sigmoid'])
    FARSEC_HP.register_hp('SVM', 'degree', np.arange(2, 10))
    FARSEC_HP.register_hp('SVM', 'gamma', np.random.uniform(0, 1.0, 20))
    FARSEC_HP.register_hp('SVM', 'coef0', np.random.uniform(0, 0.1, 20))
    FARSEC_HP.register_hp('SVM', 'tol', np.random.uniform(0, 0.1, 20))

    FARSEC_HP.register_hp('KNN', 'n_neighbors', np.arange(2, 25))
    FARSEC_HP.register_hp('KNN', 'weights', ['uniform', 'distance'])
    FARSEC_HP.register_hp('KNN', 'metric', ['minkowski', 'chebyshev'])
    FARSEC_HP.register_hp('KNN', 'p', np.arange(1, 5))

    FARSEC_HP.register_hp('NB', 'var_smoothing',
                          [10**(-i) for i in range(3, 10)])

    FARSEC_HP.register_hp('LR', 'penalty', ['l1', 'l2'])
    FARSEC_HP.register_hp('LR', 'tol', np.random.uniform(0, 0.1, 20))
    FARSEC_HP.register_hp('LR', 'C', np.arange(1, 500))

    return FARSEC_HP