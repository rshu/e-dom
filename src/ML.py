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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from random import randint, random
import numpy as np
import pandas as pd
import pdb
from utilities import _randuniform, _randchoice, _randint


def _define_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    _, indices = nbrs.kneighbors(data)
    for _ in range(0, num):
        mid = randint(0, len(data) - 1)
        nn = indices[mid, randint(1, k - 1)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random()
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
        max_features=None,
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
    assert False, "TOTO HERE"
    model = GaussianNB()
    return model, GaussianNB.__name__


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