from __future__ import print_function, division

__author__ = 'amrit'
from pandas.api.types import is_numeric_dtype, is_bool_dtype
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, KernelCenterer
from sklearn.preprocessing import QuantileTransformer, Normalizer, Binarizer, PolynomialFeatures
from utilities import _randint, _randchoice, _randuniform
from sklearn.neighbors import NearestNeighbors
import numpy as np
from random import randint, random

pd.options.mode.chained_assignment = None


def nomalization(df, methods='min-max'):
    # nomalization the preprocessed columns from dataset
    # methods contain mean, and min-max.
    for c in df.columns:
        if is_numeric_dtype(df[c]) and is_bool_dtype(df[c]) != True:
            pd.to_numeric(df[c], downcast='float')
            if methods == 'mean':
                df[c] = (df[c] - df[c].mean()) / df[c].std()
            if methods == 'min-max':
                df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df


def standard_scaler():
    scaler = StandardScaler()
    return scaler, StandardScaler.__name__


def minmax_scaler():
    scaler = MinMaxScaler()
    return scaler, MinMaxScaler.__name__


def maxabs_scaler():
    scaler = MaxAbsScaler()
    return scaler, MaxAbsScaler.__name__


## IQR parameter
def robust_scaler():
    a, b = _randint(0, 50), _randint(51, 100)
    scaler = RobustScaler(quantile_range=(a, b))
    tmp = str(a) + "_" + str(b) + "_" + RobustScaler.__name__
    return scaler, tmp


def kernel_centerer():
    scaler = KernelCenterer()
    return scaler, KernelCenterer.__name__


## Tunable parameters
def quantile_transform():
    a, b = _randint(100, 1000), _randint(1000, 1e5)
    c = _randchoice(['normal', 'uniform'])
    scaler = QuantileTransformer(
        n_quantiles=a, output_distribution=c, subsample=b)
    tmp = str(a) + "_" + str(b) + "_" + c + "_" + QuantileTransformer.__name__
    return scaler, tmp


def normalizer():
    a = _randchoice(['l1', 'l2', 'max'])
    scaler = Normalizer(norm=a)
    tmp = a + "_" + Normalizer.__name__
    return scaler, tmp


## IQR parameter
def binarize():
    a = _randuniform(0, 100)
    scaler = Binarizer(threshold=a)
    tmp = str(round(a, 4)) + "_" + Binarizer.__name__
    return scaler, tmp


def polynomial():
    a = _randint(2, 10)
    b = _randchoice([True, False])
    c = _randchoice([True, False])
    scaler = PolynomialFeatures(degree=a, interaction_only=b, include_bias=c)
    tmp = str(a) + "_" + str(b) + "_" + str(
        c) + "_" + PolynomialFeatures.__name__
    return scaler, tmp


def no_transformation():
    return no_transformation.__name__, no_transformation.__name__


def transform(df, scaler):
    if scaler == no_transformation.__name__:
        if "DataFrame" in str(type(df)):
            return df
        else:
            return pd.DataFrame(df)
    elif "DataFrame" in str(type(df)):
        df1 = pd.DataFrame(scaler.fit_transform(df[df.columns[:-1]].values))
        df1['label'] = df['label']
        # df1['loc'] = df['loc']
        return df1
    elif "array" in str(type(df)):
        df1 = pd.DataFrame(scaler.fit_transform(df))
        return df1


def transform_farsec(farsec, scaler):
    if scaler == no_transformation.__name__:
        if "DataFrame" in str(type(farsec)):
            return farsec
        else:
            return pd.DataFrame(farsec)
    elif "DataFrame" in str(type(farsec)):
        farsec1 = pd.DataFrame(
            scaler.fit_transform(farsec[farsec.columns[:-1]].values))
        farsec1['label'] = farsec['label']
        return farsec1
    elif "array" in str(type(farsec)):
        farsec1 = pd.DataFrame(scaler.fit_transform(farsec))
        return farsec1


def define_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, num):
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


def smote_balance(data_train, train_label, m=0, r=0, neighbors=0):
    pos_train = []
    neg_train = []
    for j, i in enumerate(train_label):
        if i == 1:
            pos_train.append(data_train[j])
        else:
            neg_train.append(data_train[j])
    pos_train = np.array(pos_train)
    neg_train = np.array(neg_train)

    if len(pos_train) < len(neg_train):
        pos_train = define_smote(pos_train, m, k=neighbors, r=r)
        if len(neg_train) < m:
            m = len(neg_train)
        neg_train = neg_train[np.random.choice(
            len(neg_train), m, replace=False)]
    # print(pos_train,neg_train)
    data_train1 = np.vstack((pos_train, neg_train))
    label_train = [1] * len(pos_train) + [0] * len(neg_train)
    return data_train1, label_train
