# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:43:55 2020

@author: vivin
"""
import numpy as np
import itertools
import pandas as pd
import copy
import importlib
import ast

class utils(object):
    
    @staticmethod
    def confusion_matrix_metrics(cm, labels, verbose=True):
        precision = {}
        recall = {}
        f1 = {}
        tp_sum = 0
        fp_sum = 0
        prec_sum = 0
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            precision[label] = tp/(tp+fp) if tp+fp != 0 else 0.0
            tp_sum += tp
            fp_sum +=fp
            prec_sum += precision[label]
            recall[label] = tp/(tp+fn)
            f1[label] = 2*(precision[label] * recall[label])/(precision[label] + recall[label])
            if verbose:
                print('precision score for {} is {}'.format(label, precision[label]))
                print('recall score for {} is {}'.format(label, recall[label]))
                print('f1 score for {} is {}'.format(label, f1[label]))
        precision['macro'] = np.mean(list(precision.values()))
        precision['micro'] = tp_sum/(tp_sum+fp_sum)
        if verbose:
            print('macro precision is {}'.format(precision['macro']))
            print('micro precision is {}'.format(precision['micro']))
            
    @staticmethod
    def construct_model(X_train, y_train, params, sample_weights=None):
        model_name = params['name']
        model_params = params['params']
        base_model_params = params.get('base_model_params', {})
        train_data = (X_train, y_train, sample_weights) if sample_weights is not None else (X_train, y_train)
        if model_name == 'DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(**model_params).fit(*train_data)
        elif model_name == 'SVC':
            from sklearn.svm import SVC
            model = SVC(**model_params).fit(*train_data)
        elif model_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier 
            model = KNeighborsClassifier(**model_params).fit(X_train, y_train)
        elif model_name == 'GaussianNB':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB(**model_params).fit(X_train, y_train)
        elif model_name == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**model_params).fit(*train_data)
        elif model_name == 'GradientBoostingClassifier':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**model_params).fit(*train_data)
        elif model_name == 'BaggedKNN':
            from sklearn.ensemble import BaggingClassifier
            from sklearn.neighbors import KNeighborsClassifier
            model = BaggingClassifier(KNeighborsClassifier(**base_model_params), **model_params).fit(*train_data)
        elif model_name == 'AdaBoostedTree':
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            model = AdaBoostClassifier(DecisionTreeClassifier(**base_model_params), **model_params).fit(*train_data)
        elif model_name == 'XGBoostClassifier':
            from xgboost import XGBClassifier
            model = XGBClassifier(**model_params).fit(*train_data)
        elif model_name == 'RidgeClassifier':
            from sklearn.linear_model import RidgeClassifier
            model = RidgeClassifier(**model_params).fit(*train_data)
        elif model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**model_params).fit(*train_data)
        elif model_name == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis(**model_params).fit(X_train, y_train)
        elif model_name == 'QDA':
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            model = QuadraticDiscriminantAnalysis(**model_params).fit(X_train, y_train)
        elif model_name == 'MLP':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(**model_params).fit(X_train, y_train)
        elif model_name == 'NearestCentroid':
            from sklearn.neighbors import NearestCentroid
            model = NearestCentroid(**model_params).fit(X_train, y_train)
        elif model_name == 'RadiusNeighborsClassifier':
            from sklearn.neighbors import RadiusNeighborsClassifier
            model = RadiusNeighborsClassifier(**model_params).fit(X_train, y_train)
        else:
            raise ValueError("unknown ML model passed in model_name")
        return model
    
    @staticmethod
    def params_iterator(params):
        param_keys = list(params.keys())
        out_arr = [dict(zip(param_keys,x)) for x in itertools.product(*params.values())]
        return out_arr
    
    @staticmethod
    def create_event_packet_generic(hist_events_dict, np_dt):
        df_arr = []
        for ticker, he_obj in hist_events_dict.items():
            ticker_data = he_obj.generate_event_at(np_dt)
            if len(ticker_data)==0:
                continue
            ticker_data['Ticker'] = ticker
            ticker_data.reset_index(inplace=True)
            ticker_data.rename(columns={'Adj Close': 'Price', 'Date': 'TimeStamp'}, inplace=True)
            df_arr.append(ticker_data)
        events = pd.concat(df_arr, axis=0)
        return events
    
    @staticmethod
    def create_event_packet_adj_ohlcv(hist_events_dict, np_dt):
        cols_to_rename = ['Open', 'High', 'Low', 'Close']
        cols_rename = dict(zip(cols_to_rename, ['Adj {}'.format(x) for x in cols_to_rename]))
        cols_rename['Date'] = 'TimeStamp'
        df_arr = []
        for ticker, he_obj in hist_events_dict.items():
            ticker_data = he_obj.generate_event_at(np_dt)
            if len(ticker_data)==0:
                continue
            ticker_data['Ticker'] = ticker
            ticker_data.reset_index(inplace=True)
            adj_factor = ticker_data['Adj Close'].values[0]/ticker_data['Close'].values[0]
            ticker_data[['Open', 'High', 'Low', 'Close']] = ticker_data[['Open', 'High', 'Low', 'Close']] * adj_factor
            ticker_data.drop(columns=['Adj Close'], inplace=True)
            ticker_data.rename(columns=cols_rename, inplace=True)
            df_arr.append(ticker_data)
        events = pd.concat(df_arr, axis=0)
        return events
        
    @staticmethod
    def create_consolidated_events_adj_ohlcv(hist_events_dict):
        cols_to_rename = ['Open', 'High', 'Low', 'Close']
        cols_rename = dict(zip(cols_to_rename, ['Adj {}'.format(x) for x in cols_to_rename]))
        cols_rename['Date'] = 'TimeStamp'
        df_arr = []
        for ticker, he_obj in hist_events_dict.items():
            ticker_data = he_obj.df.copy()
            if len(ticker_data)==0:
                continue
            ticker_data['Ticker'] = ticker
            ticker_data.reset_index(inplace=True)
            adj_factors = ticker_data['Adj Close'].values/ticker_data['Close'].values
            ticker_data[['Open', 'High', 'Low', 'Close']] = ticker_data[['Open', 'High', 'Low', 'Close']] * np.full((4, len(adj_factors)), adj_factors).T
            ticker_data.drop(columns=['Adj Close'], inplace=True)
            ticker_data.rename(columns=cols_rename, inplace=True)
            df_arr.append(ticker_data)
        events = pd.concat(df_arr, axis=0)
        return events
    
    @staticmethod
    def create_consolidated_events_generic(hist_events_dict):
        df_arr = []
        for ticker, he_obj in hist_events_dict.items():
            ticker_data = he_obj.df.copy()
            if len(ticker_data)==0:
                continue
            ticker_data['Ticker'] = ticker
            ticker_data.reset_index(inplace=True)
            ticker_data.rename(columns={'Adj Close': 'Price', 'Date': 'TimeStamp'}, inplace=True)
            df_arr.append(ticker_data)
        events = pd.concat(df_arr, axis=0)
        return events
    
    @staticmethod
    def generate_bar_data(tickers, start, end, bar_method='all'):
        from lib.engines.historical_events import HISTORICAL_EVENTS
        all_times = None
        hist_events_dict = {}
        for ticker in tickers:
            he_obj = HISTORICAL_EVENTS(ticker, start, end, method=bar_method)
            hist_events_dict[ticker] = he_obj
            all_times = he_obj.get_all_times() if all_times is None else np.union1d(all_times, he_obj.get_all_times())
        
        return all_times, hist_events_dict
    
    @staticmethod
    def str2dict(inp_str):
        """
        we assume input is in the following format:-
        key1:val1;key2:val2...
        """
        #split by commas
        if inp_str is None:
            return {}
        keyvalpairs = inp_str.split(";")
        out = {}
        for keyvalpair in keyvalpairs:
            key = keyvalpair.split(":")[0]
            try:
                val = ast.literal_eval(keyvalpair.split(":")[1])
            except:
                val = keyvalpair.split(":")[1]
            out[key] = val
        return out