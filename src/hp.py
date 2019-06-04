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

import pandas as pd
from ML import SMOTE, DT, RF, SVM, KNN, LR
from utilities import get_score
from collections import namedtuple
import random
import pdb
import numpy as np


class Hyperparameter:
    def __init__(self, preprocessing_names, learner_names):
        self.pres = preprocessing_names  # [SMOTE, ...]
        self.learns = learner_names  # ['DT', 'RF', ...]
        self.hp_names = dict(
        )  # {'SMOTE':['m','k','r'], 'RF':['n_estimators',...]}
        self.ranges = dict()

    def register_hp(self, alg, name, lst):
        if alg not in self.hp_names:
            self.hp_names[alg] = [name]
        else:
            self.hp_names[alg].append(name)

        self.ranges[f'{alg}_{name}'] = lst

    def get_rnd_hp_without_range(self):
        # pdb.set_trace()
        res_HP = dict()

        pre = random.choice(self.pres)
        res_HP[pre] = dict()
        for hp in self.hp_names[pre]:
            res_HP[pre][hp] = random.choice(self.ranges[f'{pre}_{hp}'])

        learner = random.choice(self.learns)
        res_HP[learner] = dict()
        for hp in self.hp_names[learner]:
            res_HP[learner][hp] = random.choice(self.ranges[f'{learner}_{hp}'])

        return res_HP, pre, learner
