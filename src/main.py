__author__ = 'rshu'

import os
cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]

import pandas as pd
from ML import SMOTE, DT, RF, SVM, KNN, LR
from utilities import get_score

import pdb


def demo(dataset):
    HP = {
        'SMOTE': {
            'm': 100,
            'k': 5,
            'r': 3
        },
        'DT': {
            'min_samples_split': 0.2,
            'criterion': 'entropy',
            'splitter': 'best'
        }
    }
    # read csv file
    train_df = pd.read_csv(f'{root}/data/FARSEC/{dataset}-train.csv').drop(
        ['id'], axis=1)
    test_df = pd.read_csv(f'{root}/data/FARSEC/{dataset}-train.csv').drop(
        ['id'], axis=1)

    train_df = SMOTE(train_df, HP)
    prediction = DT(train_df, test_df, HP)
    test_labels = test_df.label.values.tolist()

    rec = get_score('recall', prediction, test_labels, "NA")
    fp = get_score('false_alarm', prediction, test_labels, "NA")


def epsilon():
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
    demo('ambari')
