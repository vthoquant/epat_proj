# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:34:19 2020

@author: vivin
"""
from lib.utils import utils
import numpy as np

MOMENTUM_REBAL_M_c = [0.0, 1e-4, 1e-2, 1e-1, 1]
MOMENTUM_REBAL_WINDOW = [2, 5, 10, 20, 50, 100, 200]
MOMENTUM_REBAL_PARAMS = {
    'ma_window': MOMENTUM_REBAL_WINDOW,
    'M_c': MOMENTUM_REBAL_M_c
}
MOMENTUM_REBAL_OPT_CONFIGS = utils.params_iterator(MOMENTUM_REBAL_PARAMS)
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_GLOBAL_CONFIGS_1 = {
    'do_pca': [True, False],
    'predict_probab_thresh': [-1, 0.5, 0.7]
}

MULTICLASS_REBAL_GLOBAL_CONFIGS_2 = {
    'do_pca': [True, False],
    'use_sample_weights': [True, False],
    'predict_probab_thresh': [-1, 0.5, 0.7]
}

MULTICLASS_REBAL_GLOBAL_CONFIGS_3 = {
    'do_pca': [True, False],
    'predict_probab_thresh': [-1, 0.5, 0.7]
}

MULTICLASS_REBAL_PARAMS_DT = {"max_depth": [1, 2, 3, 5, 10], 'random_state': [40]}
MULTICLASS_REBAL_OPT_CONFIGS_DT = utils.params_iterator(MULTICLASS_REBAL_PARAMS_DT)
MULTICLASS_REBAL_OPT_CONFIGS_DT_MP = {'model_params':[{'name': 'DecisionTreeClassifier', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_DT]}
MULTICLASS_REBAL_OPT_CONFIGS_DT = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_2, **MULTICLASS_REBAL_OPT_CONFIGS_DT_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_SVC = {
    "C": [1, 2, 4], 
    "kernel": ['linear', 'poly', 'rbf'],
    "probability": [True]
}
MULTICLASS_REBAL_OPT_CONFIGS_SVC = utils.params_iterator(MULTICLASS_REBAL_PARAMS_SVC)
MULTICLASS_REBAL_OPT_CONFIGS_SVC_MP = {'model_params':[{'name': 'SVC', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_SVC]}
MULTICLASS_REBAL_OPT_CONFIGS_SVC = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_2, **MULTICLASS_REBAL_OPT_CONFIGS_SVC_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_KNN = {
    'n_neighbors': [5, 10, 20, 50, 100],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto'],
    'p': [1, 2]
}
MULTICLASS_REBAL_OPT_CONFIGS_KNN = utils.params_iterator(MULTICLASS_REBAL_PARAMS_KNN)
MULTICLASS_REBAL_OPT_CONFIGS_KNN_MP = {'model_params':[{'name': 'KNN', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_KNN]}
MULTICLASS_REBAL_OPT_CONFIGS_KNN = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_1, **MULTICLASS_REBAL_OPT_CONFIGS_KNN_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_RFC = {
    'n_estimators': [10, 30, 100, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2, 3],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_RFC = utils.params_iterator(MULTICLASS_REBAL_PARAMS_RFC)
MULTICLASS_REBAL_OPT_CONFIGS_RFC_MP = {'model_params':[{'name': 'RandomForestClassifier', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_RFC]}
MULTICLASS_REBAL_OPT_CONFIGS_RFC = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_2, **MULTICLASS_REBAL_OPT_CONFIGS_RFC_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_TREE_BOOSTED_BASE = {
    "max_depth": [1, 2, 5]
}
MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED_BASE = utils.params_iterator(MULTICLASS_REBAL_PARAMS_TREE_BOOSTED_BASE)
MULTICLASS_REBAL_PARAMS_TREE_BOOSTED_PARENT = {
    'n_estimators': [10, 30, 100, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED_PARENT = utils.params_iterator(MULTICLASS_REBAL_PARAMS_TREE_BOOSTED_PARENT)
MULTICLASS_REBAL_PARAMS_TREE_BOOSTED = {
    'params': MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED_PARENT,
    'base_model_params': MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED_BASE
}
MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED = utils.params_iterator(MULTICLASS_REBAL_PARAMS_TREE_BOOSTED)
MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED = [{'use_sample_weights': True, 'model_params':{**x, **{'name': 'AdaBoostedTree'}}} for x in MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED] + [{'use_sample_weights': False, 'model_params':{**x, **{'name': 'AdaBoostedTree'}}} for x in MULTICLASS_REBAL_OPT_CONFIGS_TREE_BOOSTED]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_KNN_BAGGED_BASE = {
    'n_neighbors': [5, 10, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto'],
    'p': [1, 2]
}
MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED_BASE = utils.params_iterator(MULTICLASS_REBAL_PARAMS_KNN_BAGGED_BASE)
MULTICLASS_REBAL_PARAMS_KNN_BAGGED_PARENT = {
    'max_samples': [0.1, 0.2, 0.5],
    'max_features': [0.1, 0.2, 0.5],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED_PARENT = utils.params_iterator(MULTICLASS_REBAL_PARAMS_KNN_BAGGED_PARENT)
MULTICLASS_REBAL_PARAMS_KNN_BAGGED = {
    'params': MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED_PARENT,
    'base_model_params': MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED_BASE
}
MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED = utils.params_iterator(MULTICLASS_REBAL_PARAMS_KNN_BAGGED)
MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED = [{'model_params':{**x, **{'name': 'BaggedKNN'}}} for x in MULTICLASS_REBAL_OPT_CONFIGS_KNN_BAGGED]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_GBC = {
    'n_estimators': [10, 30, 100],
    'loss': ['deviance'],
    'learning_rate': [0.01, 0.1],
    'max_depth': [1, 2, 3],
    'subsample': [0.2, 0.5],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_GBC = utils.params_iterator(MULTICLASS_REBAL_PARAMS_GBC)
MULTICLASS_REBAL_OPT_CONFIGS_GBC = [{'use_sample_weights': True, 'model_params':{'name': 'GradientBoostingClassifier', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_GBC] + [{'use_sample_weights': False, 'model_params':{'name': 'GradientBoostingClassifier', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_GBC]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


MULTICLASS_REBAL_PARAMS_XGB = {
    'n_estimators': [10, 30, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [1, 2, 3],
    'subsample': [0.2, 0.5],
    'objective': ["multi:softprob"],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_XGB = utils.params_iterator(MULTICLASS_REBAL_PARAMS_XGB)
MULTICLASS_REBAL_OPT_CONFIGS_XGB_MP = {'model_params':[{'name': 'XGBoostClassifier', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_XGB]}
MULTICLASS_REBAL_OPT_CONFIGS_XGB = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_2, **MULTICLASS_REBAL_OPT_CONFIGS_XGB_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_RIDGE = {
    'alpha': [1, 2, 3],
    'fit_intercept': [True, False],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_RIDGE = utils.params_iterator(MULTICLASS_REBAL_PARAMS_RIDGE)
MULTICLASS_REBAL_OPT_CONFIGS_RIDGE_MP = {'model_params': [{'name': 'RidgeClassifier', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_RIDGE]}
MULTICLASS_REBAL_OPT_CONFIGS_RIDGE = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_2, **MULTICLASS_REBAL_OPT_CONFIGS_RIDGE_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_LOGISTICREG = {
    'C': [0.5, 1, 2],
    'multi_class': ['multinomial'],
    'random_state': [40]
}
MULTICLASS_REBAL_OPT_CONFIGS_LOGISTICREG = utils.params_iterator(MULTICLASS_REBAL_PARAMS_LOGISTICREG)
MULTICLASS_REBAL_OPT_CONFIGS_LOGISTICREG_MP = {'model_params': [{'name': 'LogisticRegression', 'params': x} for x in MULTICLASS_REBAL_OPT_CONFIGS_LOGISTICREG]}
MULTICLASS_REBAL_OPT_CONFIGS_LOGISTICREG = utils.params_iterator({**MULTICLASS_REBAL_GLOBAL_CONFIGS_2, **MULTICLASS_REBAL_OPT_CONFIGS_LOGISTICREG_MP})
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_LDA = {
    'solver': ['svd', 'lsqr', 'eigen'],
}
MULTICLASS_REBAL_OPT_CONFIGS_LDA = utils.params_iterator(MULTICLASS_REBAL_PARAMS_LDA)
MULTICLASS_REBAL_OPT_CONFIGS_LDA = [{'model_params':{'name': 'LDA', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_LDA]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_QDA = {
    'reg_param': [0.0, 0.1, 0.2],
}
MULTICLASS_REBAL_OPT_CONFIGS_QDA = utils.params_iterator(MULTICLASS_REBAL_PARAMS_QDA)
MULTICLASS_REBAL_OPT_CONFIGS_QDA = [{'model_params':{'name': 'QDA', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_QDA]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""


MULTICLASS_REBAL_PARAMS_MLP = {
    'hidden_layer_sizes': [5, 10, 50, 100],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'random_state': [40]
}

MULTICLASS_REBAL_OPT_CONFIGS_MLP = utils.params_iterator(MULTICLASS_REBAL_PARAMS_MLP)
MULTICLASS_REBAL_OPT_CONFIGS_MLP = [{'model_params':{'name': 'MLP', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_MLP]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_NCC = {
    'metric': ['euclidean'],
}
MULTICLASS_REBAL_OPT_CONFIGS_NCC = utils.params_iterator(MULTICLASS_REBAL_PARAMS_NCC)
MULTICLASS_REBAL_OPT_CONFIGS_NCC = [{'model_params':{'name': 'NearestCentroid', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_NCC]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

MULTICLASS_REBAL_PARAMS_RNC = {
    'radius': [0.5, 1, 2],
    'weights': ['uniform', 'distance'],
}
MULTICLASS_REBAL_OPT_CONFIGS_RNC = utils.params_iterator(MULTICLASS_REBAL_PARAMS_RNC)
MULTICLASS_REBAL_OPT_CONFIGS_RNC = [{'X_transform': 'QuantileTransformer', 'model_params':{'name': 'RadiusNeighborsClassifier', 'params': x}} for x in MULTICLASS_REBAL_OPT_CONFIGS_RNC]
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

STRATEGY_OPT_CONFIG_MAP = {
    'MOMENTUM_REBAL_STRATEGY': MOMENTUM_REBAL_OPT_CONFIGS,
    'MULTICLASS_CLASSIFIER_REBAL': MULTICLASS_REBAL_OPT_CONFIGS_DT + MULTICLASS_REBAL_OPT_CONFIGS_XGB + MULTICLASS_REBAL_OPT_CONFIGS_KNN + MULTICLASS_REBAL_OPT_CONFIGS_SVC
}