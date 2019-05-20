from __future__ import print_function, division

__author__ = 'rshu'

import os

cwd = os.getcwd()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix

from itertools import product
from helper.transformation import *
from helper.utilities import _randchoice
from helper.ML import *
from collections import OrderedDict

metrics = ["gmeasure"]
data_path = os.path.join(cwd, "..", "..", "data", "FARSEC")

file_dic = {"ambari": ["ambari-train.csv", "ambari-test.csv"], \
            "camel": ["camel-train.csv", "camel-test.csv"], \
            "chromium": ["chromium-train.csv", "chromium-test.csv"], \
            "derby": ["derby-train.csv", "derby-test.csv"], \
            "wicket": ["wicket-train.csv", "wicket-test.csv"]
            }

ambari = []
camel = []
chromium = []
derby = []
wicket = []


def read_data(path):
    data = pd.read_csv(path)
    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1)
    return data


def random_forest(features, target):
    # rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                             max_depth=9, max_features=0.8, max_leaf_nodes=23,
    #                             min_impurity_decrease=0.0, min_impurity_split=None,
    #                             min_samples_leaf=3, min_samples_split=4,
    #                             min_weight_fraction_leaf=0.0, n_estimators=108, n_jobs=None,
    #                             oob_score=False, random_state=0, verbose=0, warm_start=False)
    rf = RandomForestClassifier()
    rf.fit(features, target)
    return rf


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


def main():
    # print(data_path)
    for project in file_dic:
        for file in file_dic[project]:
            if project == "ambari":
                ambari.append(os.path.join(data_path, file))
            if project == "camel":
                camel.append(os.path.join(data_path, file))
            if project == "chromium":
                chromium.append(os.path.join(data_path, file))
            if project == "derby":
                derby.append(os.path.join(data_path, file))
            if project == "wicket":
                wicket.append(os.path.join(data_path, file))

    # derby
    # train_dataset = read_data(derby[0])
    # test_dataset = read_data(derby[1])

    train_dataset = pd.read_csv(derby[0])
    test_dataset = pd.read_csv(derby[1])

    train_dataset, test_dataset = train_dataset.iloc[:, 1:], test_dataset.iloc[:, 1:]
    train_size = train_dataset["label"].count()
    farsec = pd.concat([train_dataset, test_dataset], ignore_index=True)
    farsec['label'] = farsec['label'].apply(lambda x: 0 if x == 0 else 1)

    # print(farsec)
    # print(train_dataset)
    # print(test_dataset)
    # print(train_size)

    # global train_x
    # global test_x
    # global train_y
    # global test_y

    # train_x = train_dataset.iloc[:, :-1]
    # train_y = train_dataset['label']

    # test_x = test_dataset.iloc[:, :-1]
    # test_y = test_dataset['label']

    # rf = random_forest(train_x, train_y)
    # rf_prediction = rf.predict(test_x)
    # result_statistics(rf_prediction)

    metric = "d2h"
    final = {}
    final_auc = {}
    epsilon_value = 0.2
    dic = {}
    dic_func = {}

    preprocess = [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * 20, kernel_centerer,
                  [quantile_transform] * 200, normalizer, [binarize] * 100]
    MLs = [NB, [KNN] * 20, [RF] * 50, [DT] * 30, [LR] * 50]
    preprocess_list = unpack(preprocess)
    MLs_list = unpack(MLs)
    combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

    if epsilon_value not in final_auc.keys():
        final_auc[epsilon_value] = []
        dic[epsilon_value] = {}

    func_str_dic = {}
    func_str_counter_dic = {}
    list_value = []
    dic_auc = {}

    for i in combine:
        scaler, tmp1 = i[0]()
        model, tmp2 = i[1]()
        string1 = tmp1 + "|" + tmp2
        func_str_dic[string1] = [scaler, model]
        func_str_counter_dic[string1] = 0

    counter = 0

    while counter != 1000:
        if counter not in dic_func.keys():
            dic_func[counter] = []

        try:
            keys = [k for k, v in func_str_counter_dic.items() if v == 0]
            key = _randchoice(keys)
            scaler, model = func_str_dic[key]
            farsec_transform = transform(farsec, scaler)

            train_data, test_data = farsec_transform.iloc[:train_size, :], farsec_transform[train_size:, :]
            measurement = run_model_farsec(train_data, test_data, model, metric, training=-2)

            if all(abs(t - measurement) > epsilon_value for t in list_value):
                list_value.append(measurement)
                func_str_counter_dic[key] += 1
            else:
                func_str_counter_dic[key] += -1

            if counter not in dic[epsilon_value].keys():
                dic[epsilon_value][counter] = []
                dic_func[counter] = []

            dic[epsilon_value][counter].append(min(list_value))
            dic_auc[counter] = min(list_value)

            counter += 1
        except:
            pass


    dic1 = OrderedDict(sorted(dic_auc.items(), key=itemgetter(0))).values()
    area_under_curve = round(auc(list(range(len(dic1))), dic1), 3)
    final[epsilon_value] = dic_auc
    final_auc[epsilon_value].append(area_under_curve)

    print(final)
    print(dic)
    print(dic_func)

if __name__ == '__main__':
    main()
