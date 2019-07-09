import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.neighbors import NearestNeighbors
from random import randint, random

import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]

train_df = pd.read_csv(f'{root}/data/FARSEC/ambari-train.csv').drop(
        ['id'], axis=1)
test_df = pd.read_csv(f'{root}/data/FARSEC/ambari-test.csv').drop(
        ['id'], axis=1)

# print(train_df)
# print(test_df)

def my_smote(data, num, k=5, r=1):
    corpus = []
    if len(data) < k:
        k = len(data) - 1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', p=r).fit(data)
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


def balance(data_train, train_label, m=0, r=0, neighbors=0):
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
        pos_train = my_smote(pos_train, m, k=neighbors, r=r)
        if len(neg_train) < m:
            m = len(neg_train)
        neg_train = neg_train[np.random.choice(len(neg_train), m, replace=False)]
    # print(pos_train,neg_train)
    data_train1 = np.vstack((pos_train, neg_train))
    label_train = [1] * len(pos_train) + [0] * len(neg_train)
    return data_train1, label_train

def naive_bayes(features, target):
    nb = GaussianNB()
    # nb = MultinomialNB()
    # nb = ComplementNB()
    # nb = BernoulliNB()
    nb.fit(features, target)
    return nb

def result_statistics(predictions):
    # print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    # print("Confusion matrix", confusion_matrix(test_y, predictions))
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    print("TN, FP, FN, TP: ", (tn, fp, fn, tp))
    # cm_lr = metrics.confusion_matrix(test_y, predictions)
    # print("Confusion matrix")
    # print(pd.DataFrame(cm_lr))

    PD = tp / (tp + fn)
    PF = fp / (fp + tn)
    PREC = tp / (tp + fp)
    F_MEASURE = 2 * PD * PREC / (PD + PREC)
    G_MEASURE = 2 * PD * (1 - PF) / (PD + 1 - PF)

    print("pd: ", PD)
    print("pf: ", PF)
    print("prec: ", PREC)
    print("f-measure: ", F_MEASURE)
    print("g-measure: ", G_MEASURE)



train_x = train_df.iloc[:, :-1]
train_y = train_df.iloc[:, -1:]
# train_y = train_dataset['label']

test_x = test_df.iloc[:, :-1]
test_y = test_df.iloc[:, -1:]

# print(train_x)

# lab = [y for x in train_y.values.tolist() for y in x]

# print("Process train_x and train_y with SMOTE balancing")
# train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=256, r=3, neighbors=15)


print("")
print("---------- Default Naive Bayes with StandardScaler----------")

standard_scaler = preprocessing.StandardScaler().fit(train_x)
print(standard_scaler)

train_x_standard_scaled = standard_scaler.transform(train_x)
test_x_standard_scaled = standard_scaler.transform(test_x)

nb_standardScaler = naive_bayes(train_x_standard_scaled, train_y)
nb_standardScaler_predictions = nb_standardScaler.predict(test_x_standard_scaled)
result_statistics(nb_standardScaler_predictions)

print("")
print("---------- Default Naive Bayes with MinMaxScaler----------")

min_max_scaler = preprocessing.MinMaxScaler()
print(min_max_scaler)

train_x_min_max_scaler = min_max_scaler.fit_transform(train_x)
test_x_min_max_scaler = min_max_scaler.transform(test_x)

nb_min_max_scaler = naive_bayes(train_x_min_max_scaler, train_y)
nb_min_max_scaler_predictions = nb_min_max_scaler.predict(test_x_min_max_scaler)
result_statistics(nb_min_max_scaler_predictions)


print("")
print("---------- Default Naive Bayes with MaxAbsScaler----------")

max_abs_scaler = preprocessing.MaxAbsScaler()
print(max_abs_scaler)

train_x_max_abs_scaler = max_abs_scaler.fit_transform(train_x)
test_x_max_abs_scaler = max_abs_scaler.transform(test_x)

nb_max_abs_scaler = naive_bayes(train_x_max_abs_scaler, train_y)
nb_max_abs_scaler_predictions = nb_max_abs_scaler.predict(test_x_max_abs_scaler)
result_statistics(nb_max_abs_scaler_predictions)


print("")
print("---------- Default Naive Bayes with RobustScaler----------")

robust_scaler = preprocessing.RobustScaler().fit(train_x)
print(robust_scaler)

train_x_robust_scaler = robust_scaler.transform(train_x)
test_x_robust_scaler = robust_scaler.transform(test_x)

nb_robust_scaler = naive_bayes(train_x_robust_scaler, train_y)
nb_robust_scaler_predictions = nb_robust_scaler.predict(test_x_robust_scaler)
result_statistics(nb_robust_scaler_predictions)


print("")
print("---------- Default Naive Bayes with Quantile Transformation----------")

quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
print(quantile_transformer)

X_train_trans = quantile_transformer.fit_transform(train_x)
X_test_trans = quantile_transformer.transform(test_x)

nb_quantile_transform = naive_bayes(X_train_trans, train_y)
nb_quantile_transform_predictions = nb_quantile_transform.predict(X_test_trans)
result_statistics(nb_quantile_transform_predictions)
