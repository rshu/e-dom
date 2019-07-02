import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.neighbors import NearestNeighbors
from random import randint, random

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

lab = [y for x in train_y.values.tolist() for y in x]

print("Process train_x and train_y with SMOTE balancing")
train_balanced_x, train_balanced_y = balance(train_x.values, lab, m=256, r=3, neighbors=15)

print("")
print("---------- Default Naive Bayes----------")

nb = naive_bayes(train_x, train_y)
# print("Trained model:", nb)

nb_predictions = nb.predict(test_x)
# print("Train Accuracy :: ", accuracy_score(train_y, nb.predict(train_x)))
result_statistics(nb_predictions)
