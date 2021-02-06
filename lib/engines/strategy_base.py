# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:10:14 2020

@author: vivin
"""
import abc
import numpy as np
import importlib

class STRATEGY_BASE(object, metaclass=abc.ABCMeta):
    trading_days_per_year = 252
    def __init__(self, identifier, initial_capital, run_days):
        self.identifier = identifier
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.run_days = run_days
    
    @abc.abstractmethod
    def update_indicators(self):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def generate_signal(self):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def square_off(self):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def place_orders(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def db_write_mkt(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def db_read_mkt(self, **kwargs):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def db_write_algo(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def update_essential_data(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def compute_perf_metrics(self):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def skip_event(self, *args, **kwargs):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def state_write(self):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def state_read(self):
        raise NotImplementedError()
        
    @staticmethod
    def compute_perf_metrics_base(trade_returns, total_trading_days):
        trade_returns = trade_returns.dropna() #remove all days where there were no trades
        avg_duration_days = total_trading_days/len(trade_returns)
        annualize_scaling = np.sqrt(STRATEGY_BASE.trading_days_per_year/avg_duration_days)
        number_positive = len(np.where(trade_returns > 0)[0])
        number_negative = len(np.where(trade_returns < 0)[0])
        avg_positive = np.mean(trade_returns.iloc[np.where(trade_returns > 0)])
        avg_negative = np.mean(trade_returns.iloc[np.where(trade_returns < 0)])
        if (number_positive + number_negative) > 0:
            hit_ratio = number_positive/(number_positive+number_negative)
            hit_ratio_norm = (number_positive * avg_positive)/((number_positive * avg_positive) - (number_negative * avg_negative))
        else:
            hit_ratio = 0.0
            hit_ratio_norm = 0.0
        cumul_returns = (trade_returns + 1).cumprod()[-1] - 1
        #cumul_returns = (trade_returns.iloc[np.where(trade_returns != 0)] + 1).cumprod()[-1] - 1
        annual_returns = ((cumul_returns + 1) ** (STRATEGY_BASE.trading_days_per_year/total_trading_days) - 1)
        sharpe_ratio = np.mean(trade_returns) * annualize_scaling/np.std(trade_returns)
        sortino_ratio = np.mean(trade_returns) * annualize_scaling/np.std(trade_returns.iloc[np.where(trade_returns < 0)])
        metrics = {
            'hit_ratio': hit_ratio,
            'hit_ratio_norm': hit_ratio_norm,
            'avg_positive': avg_positive,
            'avg_negative': avg_negative,
            'cumul_returns': cumul_returns,
            'annual_returns': annual_returns,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
        }
        return(metrics)
    
class STRATEGY_ML(STRATEGY_BASE, metaclass=abc.ABCMeta):
    def __init__(self, identifier, initial_capital, run_days, X_transform, model_params):
        super(STRATEGY_ML, self).__init__(identifier, initial_capital, run_days)
        self.training_data = None
        self.validation_data = None
        self.model = None
        self.model_params = model_params or {'name': 'DecisionTreeClassifier', 'params': {'max_depth': 4}}
        self.X_transformer = getattr(importlib.import_module("sklearn.preprocessing"), X_transform)() if X_transform is not None else None
        
    @abc.abstractmethod
    def prepare_training_data(self, **kwargs):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def prepare_validation_data(self, **kwargs):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def train_model(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def validate_model(self):
        raise NotImplementedError()
        
    def _add_talib_features(self, df):
        for ticker in self.tickers:
            for col_name, data in self.talib_feature_data.items():
                data_cols = data['data_cols']
                data_args = [df['{} {}'.format(ticker, col_name)] for col_name in data_cols]
                return_results = getattr(importlib.import_module('talib.abstract'), data['name'])(*data_args, **data['kwargs'])
                if isinstance(return_results, list):
                    #multiple return values
                    for idx, return_name in enumerate(data['return']):
                        if data['filter'][idx]:
                            df.loc[:, '{} {}'.format(ticker, return_name)] = return_results[idx]
                else:
                    df.loc[:, '{} {}'.format(ticker, data['return'][0])] = return_results
    