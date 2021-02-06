# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:54:18 2021

@author: vivin
"""
from lib.engines.historical_events import HISTORICAL_EVENTS
import argparse
import warnings
import importlib
import matplotlib.pyplot as plt

def main(
        ticker=None, 
        start=None, 
        end=None,
        modes=['DEMA', 'EMA', 'KAMA', 'MA', 'TEMA', 'TRIMA', 'WMA'],
        params = {'timeperiod': 30},
        labels=None
):
    warnings.filterwarnings("ignore")
    ticker = ticker or "^NSEI"
    start = start or "2010-10-05"
    end = end or "2020-05-29"
    params = params or [{}] * len(modes)
    labels = labels or modes
        
    he_obj = HISTORICAL_EVENTS(ticker, start, end, method='Close')
    df = he_obj.df
    
    for idx, mode in enumerate(modes):
        this_params = params[idx] if isinstance(params, list) else params
        mode_vals = getattr(importlib.import_module('talib.abstract'), mode)(df['Close'], **this_params)
        df.loc[:, labels[idx]] = mode_vals
        
    df.plot()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse optimizer arguments')
    parser.add_argument('--ticker')
    parser.add_argument('--start')
    parser.add_argument('--end')
    args = parser.parse_args()
    
    main(
        ticker=args.ticker,
        start=args.start,
        end=args.end
    )