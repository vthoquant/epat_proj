# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 00:59:08 2020

@author: vivin
"""

MULTICLASS_TALIB_FEATURE_CONFIG = {
    'ROCP': {
        'name': 'ROCP',
        'data_cols': ['Close'],
        'kwargs': {'timeperiod': 1},
        'return': ['ROCP'],
        'is_feature': [False]
    },
    
    'MACD': {
        'name': 'MACD',
        'data_cols': ['Close'],
        'kwargs': {},
        'return': ['MACD', 'MACD_s', 'MACD_h'],
        'filter': [False, False, True],
    },
    
    'ROCP_s': {
        'name': 'ROCP',
        'data_cols': ['Close'],
        'kwargs': {'timeperiod': 7},
        'return': ['ROCP_s'],
    },
    
    'RSI': {
        'name': 'RSI',
        'data_cols': ['Close'],
        'kwargs': {},
        'return': ['RSI'],
    },
}

TALIB_FEATURES_CONFIG_MAP = {
    'MULTICLASS_CLASSIFIER_REBAL': MULTICLASS_TALIB_FEATURE_CONFIG 
}