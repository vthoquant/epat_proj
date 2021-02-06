# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:04:56 2020

@author: vivin
"""
from lib.engines.historical_events import HISTORICAL_EVENTS
import numpy as np
import pandas as pd
import argparse
from talib.abstract import ROCP
import matplotlib.pyplot as plt
import warnings

def main(
        tickers=None, 
        start=None, 
        end=None,
        return_period=5
):
    warnings.filterwarnings("ignore")
    tickers = tickers or ["^NSEI", "KOTAKPSUBK.NS", "INFRABEES.NS", "HDFCMFGETF.NS" ]
    start = start or "2010-10-05"
    end = end or "2020-05-29"
        
    df_arr = []
    for ticker in tickers:
        he_obj = HISTORICAL_EVENTS(ticker, start, end, method='Close')
        df = he_obj.df
        df.rename(columns={'Close': '{} Close'.format(ticker)}, inplace=True)
        df_arr.append(df)
    
    df_fin = pd.concat(df_arr, axis=1)
    df_fin.dropna(inplace=True)
    
    for ticker in tickers:
        df_fin.loc[:, '{} 1d returns'.format(ticker)] = ROCP(df_fin['{} Close'.format(ticker)], timeperiod=1)
        df_fin.loc[:, '{} bucket returns'.format(ticker)] = ROCP(df_fin['{} Close'.format(ticker)], timeperiod=return_period)
        df_fin.loc[:, '{} growth'.format(ticker)] = 1 + df_fin['{} 1d returns'.format(ticker)]
        df_fin.loc[:, '{} cumul growth'.format(ticker)] = df_fin['{} growth'.format(ticker)].cumprod()
        
    base_ticker = tickers[0]
    all_base_returns = df_fin['{} bucket returns'.format(base_ticker)].values
    opt_percentiles = np.arange(0, 101, 10)
    opt_bins = np.percentile(all_base_returns[~np.isnan(all_base_returns)], opt_percentiles)
    for ticker in tickers[1:]:    
        base_returns_for_pos = all_base_returns[np.where(df_fin['{} bucket returns'.format(ticker)] > 0)]
        base_returns_for_neg = all_base_returns[np.where(df_fin['{} bucket returns'.format(ticker)] < 0)]    
        plt.figure()
        n, bins, patches = plt.hist(x=[base_returns_for_pos, base_returns_for_neg], bins=opt_bins, histtype='barstacked')
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('{} returns'.format(base_ticker),fontsize=15)
        plt.ylabel('frequency',fontsize=15)
        plt.title('Return direction frequency for {}'.format(ticker),fontsize=15)
        plt.legend(['positive', 'negative'])
        plt.show()
    
    df_fin.dropna(inplace=True)
    all_cumul_gwth_cols = ['{} cumul growth'.format(x) for x in tickers]
    df_fin[all_cumul_gwth_cols].plot()    
    plt.show()
    all_return_cols = ['{} bucket returns'.format(x) for x in tickers]
    corr_mat = df_fin[all_return_cols].rename(columns=dict(zip(['{} bucket returns'.format(x) for x in tickers], tickers))).corr()
    print("correl matrix is:\n {}".format(corr_mat))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse optimizer arguments')
    parser.add_argument('--tickers')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--return_period', type=int)
    args = parser.parse_args()
    tickers = args.tickers.split(',') if args.tickers is not None else None
    
    main(
        tickers=tickers,
        start=args.start,
        end=args.end,
    )