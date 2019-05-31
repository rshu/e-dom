__author__ = 'rshu'

import os

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]

import pandas as pd
from ML import SMOTE, DT, RF, SVM, KNN, LR
from utilities import get_score
import numpy as np

import pdb

# initialize range for pre-processor and learners
smote = {
    'k': list(np.arange(1,20)),
    'm': list(np.arange(50, 400)),
    'r': list(np.arange(1,6))
}

dt = {
    'min_samples_split': list(np.arange(2, 20, 1)),
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

rf = {
    'n_estimators': list(np.arange(10, 150)),
    'min_samples_leaf': list(np.arange(1, 20)),
    'min_samples_split': list(np.arange(2, 20)),
    'max_leaf_nodes': list(np.arange(2, 50)),
   'max_features': list(np.arange(0.01, 1, 0.01)),
    'max_depth': list(np.arange(1, 10))
}

svm = {
   'C': list(np.arange(1, 500)),
   'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
   'degree': list(np.arange(2, 10)),
   'gamma': list(np.arange(0.0, 1.0, 0.1)),
   'coef0': list(np.arange(0.0, 0.1, 0.01)),
   'tol': list(np.arange(0.0, 0.1, 0.01)),
}

knn = {
    'n_neighbors': list(np.arange(1, 10)),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'chebyshev']
}

lr = {
    'penalty': ['l1', 'l2'],
    'tol': list(np.arange(0.0, 0.1, 0.01)),
    'C': list(np.arange(1, 500))
}

mp = {
    'alpha': list(np.arange(0.0001, 0.001, 0.0001)),
    'learning_rate_init': list(np.arange(0.001, 0.01, 0.001)),
    'power_t': list(np.arange(0.1, 1, 0.1)),
    'max_iter': list(np.arange(50, 300)),
    'momentum': list(np.arange(0.1, 1, 0.1)),
    'n_iter_no_change': list(np.arange(1, 100))
}

nb = {
}

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def demo(dataset):
    global HP
    HP = {
        'SMOTE': {
            'm': 200,
            'k': 5,
            'r': 2
        },
        'DT': {
            'min_samples_split': 0.2,
            'criterion': 'entropy',
            'splitter': 'best'
        },
        'LR': {
            'penalty': 'l1',
            'tol': 0.05,
            'C': 100
        },
        'RF': {
            'n_estimators': 100,
            'criterion': 'gini',
            'min_samples_split': 0.5
        }
    }
    # # read csv file (windows)
    # train_df = pd.read_csv("C:\\Users\\Terry\\Documents\\e-dom\\data\\FARSEC\\ambari-train.csv").drop(['id'], axis=1)
    # test_df = pd.read_csv("C:\\Users\\Terry\\Documents\\e-dom\\data\\FARSEC\\ambari-test.csv").drop(['id'], axis=1)

    # read csc file (MacBook)
    train_df = pd.read_csv(f'{root}/data/FARSEC/{dataset}-train.csv').drop(
        ['id'], axis=1)
    test_df = pd.read_csv(f'{root}/data/FARSEC/{dataset}-test.csv').drop(
        ['id'], axis=1)

    train_df = SMOTE(train_df, HP)
    prediction = RF(train_df, test_df, HP)
    test_labels = test_df.label.values.tolist()

    global rec, fpr, gm
    rec = get_score('recall', prediction, test_labels, "NA")
    fpr = get_score('false_alarm', prediction, test_labels, "NA")
    gm = get_score("g_measure", prediction, test_labels, "NA")


def epsilon():
    epsilon_value_obj1 = 0.2
    epsilon_value_obj2 = 0.2

    # Initialize an empty matrix

    cell = {
        "pd": 0.0,
        "pf": 0.0,
        "gm": 0.0,
        # "combine": "",
        "count": 0
    }

    epsilon_matrix = [[cell for i in range(int(1 / epsilon_value_obj1))] for j in
                      range(int(1 / epsilon_value_obj2))]
    print("recall:", rec)
    print("false positive rate:", fpr)
    print("g measure", gm)

    # map pd and pf into grid
    x = int(rec / epsilon_value_obj1)
    y = int(rec / epsilon_value_obj2)

    if epsilon_matrix[x][y]["count"] == 0:
        epsilon_matrix[x][y]["pd"] = rec
        epsilon_matrix[x][y]["pf"] = fpr
        epsilon_matrix[x][y]['gm'] = gm
        # epsilon_matrix[x][y]["combine"] = Merge(HP['SMOTE'], HP['RF'])
        epsilon_matrix[x][y]["count"] = 1
    elif epsilon_matrix[x][y]["count"] == 1:
        if gm > epsilon_matrix[x][y]['gm']:
            epsilon_matrix[x][y]["pd"] = rec
            epsilon_matrix[x][y]["pf"] = fpr
            epsilon_matrix[x][y]['gm'] = gm
            # epsilon_matrix[x][y]["combine"] = Merge(HP['SMOTE'], HP['RF'])
            # update range
        else:
            pass
    else:
        pass

    print(epsilon_matrix)


# print pareto frontier
def pareto():
    pass


#     farsec = pd.concat([train_df, test_df], ignore_index=True)

#     preprocess = [
#         standard_scaler, minmax_scaler, maxabs_scaler, robust_scaler,
#         kernel_centerer, quantile_transform, normalizer, binarize
#     ]
#     MLs = [NB, [KNN] * 20, [RF] * 50, [DT] * 30, [LR] * 50]
#     preprocess_list = unpack(preprocess)
#     pdb.set_trace()
#     MLs_list = unpack(MLs)
#     combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

#     if epsilon_value not in final_auc.keys():
#         final_auc[epsilon_value] = []
#         dic[epsilon_value] = {}

#     func_str_dic = {}
#     func_str_counter_dic = {}
#     list_value = []
#     dic_auc = {}

#     for i in combine:
#         scaler, tmp1 = i[0]()
#         model, tmp2 = i[1]()
#         string1 = tmp1 + "|" + tmp2
#         func_str_dic[string1] = [scaler, model]
#         func_str_counter_dic[string1] = 0

#     counter = 0

#     while counter != 10:
#         if counter not in dic_func.keys():
#             dic_func[counter] = []

#         try:
#             keys = [k for k, v in func_str_counter_dic.items() if v == 0]
#             key = _randchoice(keys)
#             scaler, model = func_str_dic[key]
#             farsec_transform = transform_farsec(farsec, scaler)

#             train_data, test_data = farsec_transform.iloc[:train_df.shape[
#                 0], :], farsec_transform.iloc[train_df.shape[0]:, :]
#             measurement = run_model_farsec(
#                 train_data, test_data, model, metric, training=-2)

#             if all(abs(t - measurement) > epsilon_value for t in list_value):
#                 list_value.append(measurement)
#                 func_str_counter_dic[key] += 1
#             else:
#                 func_str_counter_dic[key] += -1

#             if counter not in dic[epsilon_value].keys():
#                 dic[epsilon_value][counter] = []
#                 dic_func[counter] = []

#             dic[epsilon_value][counter].append(min(list_value))
#             dic_auc[counter] = min(list_value)

#             counter += 1
#         except:
#             pass

#     # dic1 = OrderedDict(sorted(dic_auc.items(), key=itemgetter(0))).values()
#     # area_under_curve = round(auc(list(range(len(dic1))), dic1), 3)
#     final[epsilon_value] = dic_auc
#     # final_auc[epsilon_value].append(area_under_curve)

#     print(final)
#     print(dic)
#     print(dic_func)

if __name__ == '__main__':
    demo('derby')
    epsilon()
