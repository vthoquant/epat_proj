# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:27:24 2020

@author: vivin
"""

import yfinance as yf
import datetime
import pandas as pd
import logging

class HISTORICAL_EVENTS(object):
    def __init__(self, ticker, start, end, interval="1d", method="Adj Close"):
        self.method = method
        if isinstance(start, str):
            start = datetime.datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")
        self.df = yf.download(ticker, start, end, interval=interval)
        self._process_df()
        
    def _process_df(self):
        if isinstance(self.method, str) and self.method in self.df.columns.values:
            self.df = self.df[[self.method]]
        elif isinstance(self.method, list) and set(self.method).issubset(set(self.df.columns.values)):
            self.df = self.df[self.method]
        elif self.method == 'all':
            logging.info('not filtering any columns')
        else:
            raise Exception("unknown method specified")
        
    def generate_event_at(self, np_dt):
        try:
            out = self.df.loc[[np_dt]]
        except:
            out = pd.DataFrame(columns=self.df.columns.values) #else reutrn empty dataframe
        return out
    
    def get_all_times(self):
        return self.df.index.values
        