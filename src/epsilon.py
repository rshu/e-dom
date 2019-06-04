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

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]

from hp import Hyperparameter
import collections
import numpy as np
import ML


def get_HP_obj():
    FARSEC_HP = Hyperparameter(
        preprocessing_names=['SMOTE'],
        # learner_names=['DT', 'RF', 'SVM', 'KNN', 'LR'])
        learner_names=['DT', 'RF'])

    # TODO fill here
    FARSEC_HP.register_hp('SMOTE', 'k', np.arange(1, 20))
    FARSEC_HP.register_hp('SMOTE', 'm', np.arange(50, 400))
    FARSEC_HP.register_hp('SMOTE', 'r', np.arange(1, 6))

    FARSEC_HP.register_hp('DT', 'min_samples_split', np.arange(2, 20, 1))
    FARSEC_HP.register_hp('DT', 'criterion', ['gini', 'entropy'])
    FARSEC_HP.register_hp('DT', 'splitter', ['best', 'random'])

    FARSEC_HP.register_hp('RF', 'n_estimators', np.arange(10, 150))
    FARSEC_HP.register_hp('RF', 'min_samples_leaf', np.arange(1, 20))
    FARSEC_HP.register_hp('RF', 'min_samples_split', np.arange(2, 20))
    # FARSEC_HP.register_hp('RF', 'max_leaf_nodes', np.arange(2, 50))
    # FARSEC_HP.register_hp('RF', 'max_features', np.arange(0.01, 1, 0.01))
    # FARSEC_HP.register_hp('RF', 'max_depth', np.arange(1, 10))

    return FARSEC_HP


# Assumping all objectives are to maximize
def epsilon(dataset, HP_obj, eval_func, N1, epsilon_lst):
    weights = dict()
    for name in HP_obj.pres + HP_obj.learns:
        weights[name] = 0

    # Step1: find the best preprocessor and best learner
    best_values = [np.NINF for _ in range(len(epsilon_lst))]

    for _ in range(N1):
        rnd_hp, pre, learner = HP_obj.get_rnd_hp_without_range()
        values = eval_func(dataset, rnd_hp)

        diff_weight = 0
        diff_weight += collections.Counter(
            (values - best_values) > epsilon_lst)[True] / len(epsilon_lst)
        diff_weight -= collections.Counter(
            (values - best_values) < -epsilon_lst)[True] / len(epsilon_lst)

        weights[pre] += diff_weight
        weights[learner] += diff_weight

        best_values = np.maximum(best_values, values)

    best_pre, best_learner = HP_obj.pres[0], HP_obj.learns[0]
    for i in HP_obj.pres:
        if weights[i] > weights[best_pre]:
            best_pre = i
    for j in HP_obj.learns:
        if weights[j] > weights[best_learner]:
            best_learner = j

    # Step2
    # TODO ...


if __name__ == '__main__':
    FARSEC_HP = get_HP_obj()
    epsilon('derby', FARSEC_HP, ML.evaluation, 100, [0.2, 0.2, 0.2])
