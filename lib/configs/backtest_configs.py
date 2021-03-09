# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:35:59 2020

@author: vivin
"""

MOMENTUM_REBAL_BT_PARAMS = {
    'strategy_name': 'MOMENTUM_REBAL_STRATEGY',
    'params': {
        'rebal_freq_days': 7,
        'ma_window': 10,
        'M_c': 0.0
    }
}

MULTICLASS_DECTREE_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7,
        'do_pca': False,
        'predict_probab_thresh': 0.5,
        'use_sample_weights': False,
        'model_params': {
            'name': 'DecisionTreeClassifier', 
            'params': {'max_depth': 10, 'random_state': 40}
        }
    }
}

MULTICLASS_SVC_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7,
        'do_pca': True,
        'use_sample_weights': True,
        'model_params': {
            'name': 'SVC', 
            'params': {"C": 2, "kernel": 'poly'}
        }
    }
}

MULTICLASS_KNN_REBAL_BT_PARAMS_D = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7,
        'do_pca': True,
        'predict_probab_thresh': 0.5,
        'model_params': {
            'name': 'KNN', 
            'params': {"n_neighbors": 10, "weights": 'distance', "p": 2}
        }
    }
}

MULTICLASS_KNN_REBAL_BT_PARAMS_U = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'do_pca': True,
        'predict_probab_thresh': 0.5,
        'model_params': {
            'name': 'KNN', 
            'params': {"n_neighbors": 10, "weights": 'uniform', "p": 2}
        }
    }
}

MULTICLASS_GNB_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {}
    }
}

MULTICLASS_KNN_REBAL_BT_PARAMS_U_BAGGED = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {
            'name': 'BaggedKNN', 
            'base_model_params': {"n_neighbors": 10, "weights": 'uniform', "p": 1},
            'params': {'max_samples': 0.5, 'max_features': 0.5, 'random_state': 40}
        }
    }
}

MULTICLASS_KNN_REBAL_BT_PARAMS_D_BAGGED = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {
            'name': 'BaggedKNN', 
            'base_model_params': {"n_neighbors": 10, "weights": 'distance', "p": 1},
            'params': {'max_samples': 0.5, 'max_features': 0.5, 'random_state': 40}
        }
    }
}

MULTICLASS_GBOOSTING_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {
            'name': 'GradientBoostingClassifier', 
            'params': {"n_estimators": 30, "loss": 'deviance', "learning_rate": 0.5, "subsample": 0.5, "max_depth": 2, 'random_state': 40},
        }
    }
}

MULTICLASS_RFOREST_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {
            'name': 'RandomForestClassifier', 
            'params': {"n_estimators": 300, "criterion": 'entropy', "max_depth": 5, 'random_state': 40},
        }
    }
}

MULTICLASS_XGBOOST_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7,
        'do_pca': False,
        'predict_probab_thresh': 0.5,
        'use_sample_weights': False,
        'model_params': {
            'name': 'XGBoostClassifier',
            'params': {"n_estimators": 100, 'objective': "multi:softprob", "learning_rate": 0.1, "subsample": 0.2, "max_depth": 3, 'random_state': 40},
        }
    }
}

MULTICLASS_ADABOOST_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {
            'name': 'AdaBoostedTree',
            'params': {"n_estimators": 30, "learning_rate": 0.5, 'random_state': 40},
        }
    }
}

MULTICLASS_MLP_REBAL_BT_PARAMS = {
    'strategy_name': 'MULTICLASS_CLASSIFIER_REBAL',
    'params':{
        'rebal_freq_days': 7, 
        'model_params': {
            'name': 'MLP',
            'params': {"hidden_layer_sizes": 10, "activation": 'relu', 'solver': 'adam', 'random_state': 401},
        }
    }
}

STRATEGY_BT_CONFIG_MAP = {
    'mom-rebal': MOMENTUM_REBAL_BT_PARAMS,
    'multiclass-dectree': MULTICLASS_DECTREE_REBAL_BT_PARAMS,
    'multiclass-svc': MULTICLASS_SVC_REBAL_BT_PARAMS,
    'multiclass-knn-d': MULTICLASS_KNN_REBAL_BT_PARAMS_D,
    'multiclass-knn-u': MULTICLASS_KNN_REBAL_BT_PARAMS_U,
    'multiclass-gnb': MULTICLASS_GNB_REBAL_BT_PARAMS,
    'multiclass-knn-u-bagged': MULTICLASS_KNN_REBAL_BT_PARAMS_U_BAGGED,
    'multiclass-knn-d-bagged': MULTICLASS_KNN_REBAL_BT_PARAMS_D_BAGGED,
    'multiclass-gboosting': MULTICLASS_GBOOSTING_REBAL_BT_PARAMS,
    'multiclass-rforest': MULTICLASS_RFOREST_REBAL_BT_PARAMS,
    'multiclass-xgb': MULTICLASS_XGBOOST_REBAL_BT_PARAMS,
    'multiclass-adaboost': MULTICLASS_ADABOOST_REBAL_BT_PARAMS,
    'multiclass-mlp': MULTICLASS_MLP_REBAL_BT_PARAMS 
}