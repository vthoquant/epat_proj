# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:31:11 2020

@author: vivin
"""

prefix = 'lib.strategies'

STRATEGY_MAP_TEMP = {
    #classname to filename mapping for strategies
    "MULTICLASS_CLASSIFIER_REBAL": "multi_class_rebal",
    "MOMENTUM_REBAL_STRATEGY": "momentum_rebal"
}

STRATEGY_MAP = {k:'{}.{}'.format(prefix,v) for k,v in STRATEGY_MAP_TEMP.items()}