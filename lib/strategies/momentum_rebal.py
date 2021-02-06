# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:23:44 2020

@author: vivin
"""

from lib.engines.strategy_base import STRATEGY_BASE
import pandas as pd
import numpy as np
import os
import json

class MOMENTUM_REBAL_STRATEGY(STRATEGY_BASE):
    db_loc = "C:\\Users\\vivin\\Documents\\QuantInsti\\project_data\\"
    excl_attr_store = ['db_cache_mkt', 'last_processed_time', 'db_cache_algo']
    def __init__(self, identifier, initial_capital=1000000, run_days=0, tickers=None, ma_window=None, M_c=None, rebal_freq_days=7):
        STRATEGY_BASE.__init__(self, identifier, initial_capital, run_days)
        self.ma_window = ma_window
        self.M_c = M_c
        self.tickers = tickers
        self.days_since_start = 0
        self.weights = dict(zip(tickers, [0.0] * len(tickers)))
        self.weights['Cash'] = 1.0
        self.per_asset_capital = dict(zip(tickers, [0.0] * len(tickers)))
        self.per_asset_capital['Cash'] = initial_capital
        self.rebal_freq_days = rebal_freq_days
        self.live_prices = dict(zip(tickers, [None] * len(tickers)))
        self.units_float = dict(zip(tickers, [0.0] * len(tickers)))
        self.units_whole = self.units_float
        self.units_whole_prev = self.units_whole
        self.per_asset_signal = dict(zip(tickers, [0.0] * len(tickers)))
        self.last_processed_time = None
        
        self.db_cache_mkt = pd.DataFrame(columns=['{} Price'.format(x) for x in tickers] + ['TimeStamp']).set_index('TimeStamp')
        price_cols = ['{} Price'.format(x) for x in tickers]
        position_cols = ['{} Position'.format(x) for x in tickers]
        return_cols = ['{} returns'.format(x) for x in tickers]
        capital_cols = ['{} capital'.format(x) for x in tickers + ['Cash']]
        self.db_cache_algo = pd.DataFrame(columns=['TimeStamp'] + price_cols + capital_cols + position_cols + return_cols + ['Eq Curve Unrealized', 'Eq Curve Realized']).set_index('TimeStamp')
        
    def generate_signal(self):
        for ticker in self.tickers:
            self.per_asset_signal[ticker] = self.units_whole[ticker] - self.units_whole_prev[ticker]
        
    def update_indicators(self):
        self.run_days = self.run_days + 1
        self.units_whole_prev = self.units_whole.copy()
        if self.run_days > self.ma_window:
            self.days_since_start = self.days_since_start + 1
            data = self.db_read_mkt(self.ma_window+1) #+1 because first return is always nan
            mom_dict = {}
            for ticker in self.tickers:
                prices_hist = data['{} Price'.format(ticker)]
                returns = prices_hist.pct_change()[1:].fillna(0)
                momentum = returns.mean()
                mom_dict[ticker] = momentum
                price = prices_hist.values[-1]
                self.live_prices[ticker] = price
                if self.run_days > self.ma_window + 1:
                    #update per asset capital based on c-c returns
                    self.per_asset_capital[ticker] = self.per_asset_capital[ticker] * (1+returns[-1])
            if self.days_since_start > 1:
                self.current_capital = np.array(list(self.per_asset_capital.values())).sum()
            if ((self.days_since_start-1) % self.rebal_freq_days) == 0:
                #rebalance 
                self._rebalance(mom_dict)
        else:
            data = self.db_read_mkt(1) #just the latest price to update live prices
            for ticker in self.tickers:
                price = data['{} Price'.format(ticker)].values[-1]
                self.live_prices[ticker] = price
    
    def _convert_units_float_to_whole(self):
        self.units_whole = self.units_float.copy()
        for ticker in self.tickers:
            self.units_whole[ticker] = int(self.units_whole[ticker])
    
    def _rebalance(self, mom_dict):
        #mom_dict['Cash'] = self.M_c
        pos_mom_sum = np.array([max(m,0) for m in mom_dict.values()]).sum() + self.M_c
        assigned_wts = 0
        for ticker, momentum in mom_dict.items():
            #if ticker != 'Cash':
            self.weights[ticker] = max(momentum,0)/pos_mom_sum if pos_mom_sum != 0.0 else 0.0
            assigned_wts = assigned_wts + self.weights[ticker]
            self.units_float[ticker] = (self.current_capital * self.weights[ticker])/self.live_prices[ticker]

        self.weights['Cash'] = 1 - assigned_wts
        self._convert_units_float_to_whole()
        allocated_capital = 0
        for ticker in self.tickers:
            #rebalance capital based on new units
            self.per_asset_capital[ticker] = self.units_whole[ticker] * self.live_prices[ticker]
            allocated_capital = allocated_capital + self.per_asset_capital[ticker]
        self.per_asset_capital['Cash'] = self.current_capital - allocated_capital
        
    def update_essential_data(self, is_square_off=False):
        live_data = self.db_read_mkt(2)
        ts = live_data.index.values[-1]
        df_price = live_data.copy()
        df_price[['{} returns'.format(x.split(" ")[0]) for x in df_price.columns.values]] = df_price[df_price.columns.values].pct_change()
        df_price = df_price.iloc[[-1]]
        df_pos = pd.DataFrame(self.units_whole, index=[ts])
        df_pos.columns = ['{} Position'.format(x) for x in df_pos.columns.values]
        df_pos.index.rename('TimeStamp', inplace=True)
        df_capital = pd.DataFrame(self.per_asset_capital, index=[ts])
        df_capital.columns = ['{} capital'.format(x) for x in df_capital.columns.values]
        df_capital.index.rename('TimeStamp', inplace=True)
        df_temp = pd.concat([df_price, df_pos, df_capital], axis=1)
        self.db_cache_algo = pd.concat([self.db_cache_algo, df_temp], axis=0)
        self.db_cache_algo.loc[ts, 'Eq Curve Unrealized'] = self.current_capital
        if (((self.days_since_start-1) % self.rebal_freq_days) == 0) or is_square_off: # it is a rebalance day
            self.db_cache_algo.loc[ts, 'Eq Curve Realized'] = self.current_capital
        
    def db_write_mkt(self, events_df, toDb=False):
        if events_df is not None:
            """this will be in a specific format"""
            MOMENTUM_REBAL_STRATEGY.check_events(events_df)
            ts = events_df['TimeStamp'].values[0]
            cols = events_df['Ticker'].values
            cols = ['{} Price'.format(x) for x in cols]
            prices = events_df['Price'].values
            df_dict = dict(zip(cols, prices))
            db_df = pd.DataFrame(df_dict, index=[0])
            db_df['TimeStamp'] = ts
            db_df = db_df.set_index('TimeStamp')        
            self.db_cache_mkt = pd.concat([self.db_cache_mkt, db_df], axis=0)
            self.last_processed_time = ts
        
        if toDb:
            #hack to mimic db write
            table_loc = self.db_loc + self.identifier + ".csv"
            if os.path.isfile(table_loc):
                df_old = pd.read_csv(table_loc)
                df_old = df_old.set_index('TimeStamp')
                df_old.index = pd.to_datetime(df_old.index)
                df_new = pd.concat([df_old[~df_old.index.isin(self.db_cache_mkt.index)], self.db_cache_mkt])
                df_new.to_csv(table_loc)
            else:
                self.db_cache_mkt.to_csv(table_loc)
    
    @staticmethod
    def check_events(events_df):
        assert len(events_df) > 0, "empty events passed!"
        assert isinstance(events_df, pd.core.frame.DataFrame), "input events is not of type DataFrame"
        assert set(['Ticker', 'TimeStamp', 'Price']).issubset(events_df.columns.values), "Ticker, timestamp and price not present in events data"
        assert len(np.unique(events_df['TimeStamp'].values)) == 1, "events have different timestamps!"
    
    def db_read_mkt(self, window=20, fromDb=False):
        if len(self.db_cache_mkt)==0 or fromDb:
            table_loc = self.db_loc + self.identifier + ".csv"
            db_table_full = pd.read_csv(table_loc)
            db_table = db_table_full.iloc[-1*min(len(db_table_full), window):]
            self.db_cache_mkt = db_table
            self.db_cache_mkt.set_index('TimeStamp', inplace=True)
            self.db_cache_mkt.index = pd.to_datetime(self.db_cache_mkt.index)
        else:
            db_table = self.db_cache_mkt.iloc[-1*min(len(self.db_cache_mkt), window):]
        return db_table
    
    def state_write(self):
        attributes_curr = self.__dict__
        attributes_curr = {k:v for k,v in attributes_curr.items() if k not in self.excl_attr_store}
        attributes_curr['last_processed_time'] = self.last_processed_time.astype(str) #special handling for numpy datetime
        file_loc = self.db_loc + self.identifier + "-attr.txt"
        with open(file_loc, 'w') as outfile:
            json.dump(attributes_curr, outfile)
    
    def state_read(self):
        file_loc = self.db_loc + self.identifier + "-attr.txt"
        with open(file_loc, 'r') as json_file:
            attributes = json.load(json_file)
        for attr in attributes:
            setattr(self, attr, attributes[attr])
        self.last_processed_time = np.datetime64(attributes['last_processed_time'])
            
    def skip_event(self, events_df):
        skip = False
        skip = skip or (len(np.unique(events_df['TimeStamp'].values)) != 1)
        skip = skip or (len(events_df) != len(self.tickers))
        return skip
        
    def place_orders(self):
        '''
        for ticker, units in self.per_asset_signal.items():
            print("{}: placing order for {} units of {}".format(self.last_processed_time, units, ticker))
            print("{}: current position in {} is {} units".format(self.last_processed_time, ticker, self.units_whole[ticker]))
        print("{}: current capital is {}".format(self.last_processed_time, self.current_capital))
        '''
        pass
            
    def db_write_algo(self):
        #hack to mimic db write
        table_loc = self.db_loc + self.identifier + "-algo.csv"
        if os.path.isfile(table_loc):
            df_old = pd.read_csv(table_loc)
            df_old = df_old.set_index('TimeStamp')
            df_old.index = pd.to_datetime(df_old.index)
            df_new = pd.concat([df_old[~df_old.index.isin(self.db_cache_algo.index)], self.db_cache_algo])
            df_new.to_csv(table_loc)
        else:
            self.db_cache_algo.to_csv(table_loc)
        
    def square_off(self):
        self.units_whole_prev = self.units_whole.copy()
        for ticker in self.tickers:
            self.units_whole[ticker] = 0
            self.per_asset_signal[ticker] = - self.units_whole_prev[ticker]
            self.per_asset_capital[ticker] = 0.0
        self.per_asset_capital['Cash'] = self.current_capital
        self.place_orders()
        self.update_essential_data(is_square_off=True)
        
    def compute_perf_metrics(self):
        """
        given all the trade returns, compute typical metrics such as hit ratio, normalized hit ratio etc
        """
        trade_returns = self.db_cache_algo['Eq Curve Realized'].dropna().pct_change().dropna()
        metrics = self.compute_perf_metrics_base(trade_returns, self.days_since_start)
        return(metrics)
        
    
