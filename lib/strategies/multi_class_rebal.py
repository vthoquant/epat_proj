# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:23:26 2020

@author: vivin
"""
from lib.engines.strategy_base import STRATEGY_ML
from lib.utils import utils
from lib.configs.talib_feature_configs import TALIB_FEATURES_CONFIG_MAP 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import json

class MULTICLASS_CLASSIFIER_REBAL(STRATEGY_ML):
    db_loc = "C:\\Users\\vivin\\Documents\\QuantInsti\\project_data\\"
    excl_attr_store = ['db_cache_mkt', 'last_processed_time', 'db_cache_algo', 'training_data', 'validation_data']
    def __init__(self, identifier, initial_capital=1000000, run_days=0, tickers=None, long_window=40, rebal_freq_days=7, X_transform=None, use_sample_weights=False, model_params=None, do_pca=True, pca_thresh=0.95, predict_probab_thresh=-1):
        super(MULTICLASS_CLASSIFIER_REBAL, self).__init__(identifier, initial_capital, run_days, X_transform, model_params)
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
        self.long_window = long_window
        self.talib_feature_data = TALIB_FEATURES_CONFIG_MAP['MULTICLASS_CLASSIFIER_REBAL']
        self.feature_col_names = self._feature_col_names()
        self.use_sample_weights = use_sample_weights
        
        self.do_pca = do_pca
        self.pca_thresh = pca_thresh
        self.pca_model = None
        self.X_scaler_transformer = None
        self.pca_comp_thresh_idx = None
        
        self.predict_probab_thresh = predict_probab_thresh
        
        cols_ohlcv = []
        for value_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
            cols_ohlcv = cols_ohlcv + ['{} {}'.format(x, value_type) for x in tickers]
        self.db_cache_mkt = pd.DataFrame(columns=cols_ohlcv + ['TimeStamp']).set_index('TimeStamp')
        position_cols = ['{} Position'.format(x) for x in tickers]
        return_cols = ['{} returns'.format(x) for x in tickers]
        capital_cols = ['{} capital'.format(x) for x in tickers + ['Cash']]
        self.db_cache_algo = pd.DataFrame(columns=['TimeStamp'] + cols_ohlcv + capital_cols + position_cols + return_cols + ['Eq Curve Unrealized', 'Eq Curve Realized']).set_index('TimeStamp')
        
    def _feature_col_names(self):
        feature_cols = []
        talib_feature_tags = []
        for talib_data in self.talib_feature_data.values():
            is_feature = talib_data.get('is_feature', talib_data.get('filter', [True] * len(talib_data['return'])))
            return_vals = talib_data['return']
            for idx, return_val in enumerate(return_vals):
                if is_feature[idx]:
                    talib_feature_tags.append(return_val)
            
        for ticker in self.tickers:
            '''
            feature_cols.append('{} returns'.format(ticker))
            feature_cols.append('{} Long MA'.format(ticker))
            feature_cols.append('{} Short MA'.format(ticker))
            feature_cols.append('{} Long risk'.format(ticker))
            feature_cols.append('{} Short risk'.format(ticker))
            '''
            feature_cols = feature_cols + ['{} {}'.format(ticker, x) for x in talib_feature_tags] #+ ['{} mom_wt'.format(ticker)] + ['{} bband_loc'.format(ticker)] 
            
        return feature_cols
    
    def prepare_training_data(self):
        self.training_data = self.db_cache_mkt.copy()
        self.training_data = self._add_features_and_labels(self.training_data)
        self.training_data.dropna(inplace=True)
        
    def prepare_validation_data(self):
        self.validation_data = self.db_cache_mkt.copy() #copy all data to this first
        self.validation_data = self.validation_data[~self.validation_data.index.isin(self.training_data.index)]
        self.validation_data = self._add_features_and_labels(self.validation_data)
        self.validation_data.dropna(inplace=True)
        
    def _add_features_and_labels(self, df):
        df = self._add_features(df)
        all_predicted_return_series = []
        for ticker in self.tickers:
            df.loc[:, '{} 1d g-factor'.format(ticker)] = 1+df['{} ROCP'.format(ticker)]
            df.loc[:, '{} 1d g-factor pred'.format(ticker)] = df['{} 1d g-factor'.format(ticker)].shift(-1)
            rebal_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.rebal_freq_days)
            predicted_return_series = df['{} 1d g-factor pred'.format(ticker)].rolling(window=rebal_indexer).apply(np.prod, raw=True)
            df.loc[:, '{} f-day g-factor pred'.format(ticker)] = predicted_return_series
            all_predicted_return_series.append(predicted_return_series)
        self._add_sample_weights(df, all_predicted_return_series)
        df['Cash f-day g-factor pred'] = 1.0
        df['winning'] = df[['{} f-day g-factor pred'.format(ticker) for ticker in self.tickers+['Cash']]].idxmax(axis=1)
        df['winning ticker'] = df['winning'].str.split(expand=True)[0]
        cols_to_drop = ['{} 1d g-factor'.format(ticker) for ticker in self.tickers] + ['winning']
        df.drop(columns=cols_to_drop, inplace=True)
        df.dropna(inplace=True)
        return df
    
    @staticmethod
    def _add_sample_weights(df, all_predicted_return_series):
        predicted_max = np.maximum.reduce(all_predicted_return_series)
        predicted_max = np.maximum(predicted_max, np.full(len(predicted_max), 1.0))
        sample_weights_arr = np.full(len(predicted_max), 0.0)
        for i in range(len(all_predicted_return_series)):
            error_risk = predicted_max - all_predicted_return_series[i]
            z_score = (error_risk - np.mean(error_risk[~np.isnan(error_risk)]))/np.std(error_risk[~np.isnan(error_risk)])
            wts = np.full(len(z_score), 1.0)
            wts = np.where(np.abs(z_score) > 1, z_score ** 2, wts)
            sample_weights_arr = sample_weights_arr + wts
            
        df.loc[:, 'sample_weights'] = sample_weights_arr
        df['sample_weights'].fillna(0.0, inplace=True)
    
    def _add_features(self, df):
        for ticker in self.tickers:
            '''
            df.loc[:, '{} returns'.format(ticker)] = df['{} Close'.format(ticker)].pct_change()
            df.loc[:, '{} Long MA'.format(ticker)] = df['{} returns'.format(ticker)].rolling(window=self.long_window).mean()
            df.loc[:, '{} Short MA'.format(ticker)] = df['{} returns'.format(ticker)].rolling(window=self.short_window).mean()
            df.loc[:, '{} Long risk'.format(ticker)] = df['{} returns'.format(ticker)].rolling(window=self.long_window).std()
            df.loc[:, '{} Short risk'.format(ticker)] = df['{} returns'.format(ticker)].rolling(window=self.short_window).std()
            '''
        
        self._add_talib_features(df)
        return df
    
    """
    #NEW FEATURES TRIED OUT FOR EXPERIMENTATION PURPOSES
    def _add_features(self, df):
        from talib.abstract import ROCP
        from talib.abstract import BBANDS
        pos_mom_sum = np.full(len(df), 0)
        for ticker in self.tickers:
            df.loc[:, '{} ret'.format(ticker)] = ROCP(df['{} Close'.format(ticker)], timeperiod=1)
            
            df.loc[:, '{} mom'.format(ticker)] = df['{} ret'.format(ticker)].rolling(window=10).mean()
            df['{} mom'.format(ticker)].fillna(0.0, inplace=True)
            pos_mom_sum = pos_mom_sum + np.maximum(df['{} mom'.format(ticker)], 0.0)
            
            df.loc[:, '{} pos_count'.format(ticker)] = np.where(df['{} ret'.format(ticker)] > 0, 1, 0)
            df.loc[:, '{} cum_pos_count'.format(ticker)] = df['{} pos_count'.format(ticker)].rolling(window=10).sum()
            uband, mband, lband = BBANDS(df['{} Close'.format(ticker)], timeperiod=10)
            df.loc[:, '{} bband_loc'.format(ticker)] = np.full(len(df), 0)
            df.loc[:, '{} bband_loc'.format(ticker)] = np.where(df['{} Close'.format(ticker)] > uband, 1, df['{} bband_loc'.format(ticker)])
            df.loc[:, '{} bband_loc'.format(ticker)] = np.where(df['{} Close'.format(ticker)] < lband, -1, df['{} bband_loc'.format(ticker)])
        #df.loc[:, 'returns_correl'] = CORREL(df['^NSEI mom_wt'], df['HDFCMFGETF.NS mom_wt'])
        
        for ticker in self.tickers:
            df.loc[:, '{} mom_wt'.format(ticker)] = np.maximum(df['{} mom'.format(ticker)], 0.0)/pos_mom_sum
            df['{} mom_wt'.format(ticker)].fillna(0, inplace=True)
        
        self._add_talib_features(df)
        return df
    """
        
    def train_model(self):
        X_train = self.training_data[self.feature_col_names]
        y_train = self.training_data['winning ticker']
        sample_wts = self.training_data['sample_weights'] if self.use_sample_weights else None
        X_train = self._transform_features_set_train(X_train)
        self.model = utils.construct_model(X_train, y_train, self.model_params, sample_wts)
        y_pred_ins = self.model.predict(X_train)
        cm = confusion_matrix(y_train, y_pred_ins, sample_weight=sample_wts, labels=self.tickers+['Cash'])
        utils.confusion_matrix_metrics(cm, self.tickers+['Cash'])
        self.training_data.loc[:, 'winning ticker pred'] = y_pred_ins
        
    def validate_model(self):
        X_val = self.validation_data[self.feature_col_names]
        y_val = self.validation_data['winning ticker']
        sample_wts = self.validation_data['sample_weights'] if self.use_sample_weights else None
        X_val = self._transform_features_set(X_val)
        y_pred_val = self.model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred_val, sample_weight=sample_wts, labels=self.tickers+['Cash'])
        utils.confusion_matrix_metrics(cm, self.tickers+['Cash'])
        self.validation_data.loc[:, 'winning ticker pred'] = y_pred_val
        
    def generate_scores(self, mode='insample'):
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
        df = self.training_data if mode == 'insample' else self.validation_data
        labels = self.tickers + ['Cash']
        y_true = df['winning ticker']
        y_pred = df['winning ticker pred']
        sample_wts = df['sample_weights'] if self.use_sample_weights else None
        acc_score = accuracy_score(y_true, y_pred, sample_weight=sample_wts)
        bal_acc_score = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_wts)
        prec_score = precision_score(y_true, y_pred, labels=labels, average=None, sample_weight=sample_wts)
        recall_score = recall_score(y_true, y_pred, labels=labels, average=None, sample_weight=sample_wts)
        all_metrics = {
            '{} acc_score'.format(mode): acc_score,
            '{} bal_acc_score'.format(mode): bal_acc_score,
            '{} prec_score'.format(mode): dict(zip(labels, prec_score)),
            '{} recall_score'.format(mode): dict(zip(labels, recall_score)),
        }
        
        return all_metrics
    
    def generate_signal(self):
        for ticker in self.tickers:
            self.per_asset_signal[ticker] = self.units_whole[ticker] - self.units_whole_prev[ticker]
        
    def update_indicators(self):
        #at this point we are assuming we already have a calibrated model
        self.run_days = self.run_days + 1
        self.units_whole_prev = self.units_whole.copy()
        if self.run_days > self.long_window: #need upto this many days to be able to generate all metrics
            self.days_since_start = self.days_since_start + 1
            data = self.db_read_mkt(self.long_window+1) #+1 because first return is always nan
            for ticker in self.tickers:
                prices_hist = data['{} Close'.format(ticker)]
                returns = prices_hist.pct_change()[1:].fillna(0)
                price = prices_hist.values[-1]
                self.live_prices[ticker] = price
                if self.run_days > self.long_window + 1:
                    #update per asset capital based on c-c returns
                    self.per_asset_capital[ticker] = self.per_asset_capital[ticker] * (1+returns[-1])
            if self.days_since_start > 1:
                self.current_capital = np.array(list(self.per_asset_capital.values())).sum()
            if ((self.days_since_start-1) % self.rebal_freq_days) == 0:
                #get model output and rebalance
                self._predict_from_data(data)
                self._rebalance()
        else:
            data = self.db_read_mkt(1) #just the latest price to update live prices
            for ticker in self.tickers:
                price = data['{} Close'.format(ticker)].values[-1]
                self.live_prices[ticker] = price
                
    def _rebalance(self):
        assigned_wts = 0
        for ticker in self.tickers:
            weight = self.weights[ticker]
            assigned_wts = assigned_wts + weight
            self.units_float[ticker] = (self.current_capital * weight)/self.live_prices[ticker]

        self.weights['Cash'] = 1 - assigned_wts
        self._convert_units_float_to_whole()
        allocated_capital = 0
        for ticker in self.tickers:
            #rebalance capital based on new units
            self.per_asset_capital[ticker] = self.units_whole[ticker] * self.live_prices[ticker]
            allocated_capital = allocated_capital + self.per_asset_capital[ticker]
        self.per_asset_capital['Cash'] = self.current_capital - allocated_capital
        
    def _convert_units_float_to_whole(self):
        self.units_whole = self.units_float.copy()
        for ticker in self.tickers:
            self.units_whole[ticker] = int(self.units_whole[ticker])
            
    def _predict_from_data(self, data):
        model_data_X = self._add_features(data)
        model_data_X = model_data_X.iloc[[-1]][self.feature_col_names] #only need the last row
        model_data_X = self._transform_features_set(model_data_X)
        model_data_y = self.model.predict(model_data_X)
        predict_probab = 1.0 if self.predict_probab_thresh < 0 else np.max(self.model.predict_proba(model_data_X)[0])
        self._generate_weights(model_data_y, predict_probab)
        
    def _transform_features_set(self, X_data):
        if self.do_pca:
            X_norm = self.X_scaler_transformer.transform(X_data)
            X_data_full = self.pca_model.transform(X_norm)
            X_data = X_data_full[:, 0:self.pca_comp_thresh_idx+1]
        elif self.X_transformer is not None:
            X_data = self.X_transformer.transform(X_data)
        return X_data
    
    def _transform_features_set_train(self, X_train):
        if self.do_pca:
            self.X_scaler_transformer = StandardScaler()
            X_train_norm = self.X_scaler_transformer.fit_transform(X_train)
            #do pca on normalized data
            self.pca_model = PCA()
            X_train_full = self.pca_model.fit_transform(X_train_norm)
            self.pca_comp_thresh_idx = np.where(self.pca_model.explained_variance_ratio_.cumsum() >= self.pca_thresh)[0][0]
            X_train = X_train_full[:, 0:self.pca_comp_thresh_idx+1]
        elif self.X_transformer is not None:
            X_train = self.X_transformer.fit_transform(X_train)
        return X_train
    
    def _generate_weights(self, y, predict_probab):
        winning_ticker = 'Cash' if predict_probab < self.predict_probab_thresh else y[0]
        self.weights = dict(zip(self.tickers, [0.0] * len(self.tickers)))
        self.weights[winning_ticker] = 1.0
        
    def update_essential_data(self, is_square_off=False):
        live_data = self.db_read_mkt(2)
        ts = live_data.index.values[-1]
        df_price = live_data.copy()
        #df_price[['{} returns'.format(x.split(" ")[0]) for x in df_price.columns.values]] = df_price[df_price.columns.values].pct_change()
        df_price[['{} returns'.format(x) for x in self.tickers]] = df_price[['{} Close'.format(x) for x in self.tickers]].pct_change()
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
            MULTICLASS_CLASSIFIER_REBAL.check_events(events_df)
            ts = events_df['TimeStamp'].values[0]
            tickers = events_df['Ticker'].values
            cols = []
            values = []
            for value_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
                cols = cols + ['{} {}'.format(x, value_type) for x in tickers]
                values = values + events_df['Adj {}'.format(value_type) if value_type != 'Volume' else 'Volume'].values.tolist()
            df_dict = dict(zip(cols, values))
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
        assert set(['Ticker', 'TimeStamp']).issubset(events_df.columns.values), "Ticker and timestamp not present in events data"
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