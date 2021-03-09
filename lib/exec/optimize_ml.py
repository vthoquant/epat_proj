# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:41:21 2020

@author: vivin
"""

from lib.utils import utils
from lib.configs.optimizer_configs import STRATEGY_OPT_CONFIG_MAP
from lib.configs.strategy_mapper import STRATEGY_MAP
import importlib
import pandas as pd
import numpy as np
import argparse
import warnings
from flatten_dict import flatten

def main(
        run_name="test", 
        strategy_name='MULTICLASS_CLASSIFIER_REBAL', 
        tickers=None, 
        start=None, 
        end=None, 
        initial_capital=100000, 
        tvt_ratio=[0.6, 0.2, 0.2],
        db_loc=None
):
    warnings.filterwarnings("ignore")
    tickers = tickers or ["^NSEI", "KOTAKPSUBK.NS", "INFRABEES.NS", "HDFCMFGETF.NS" ]
    start = start or "2010-10-05"
    end = end or "2020-05-29"
    full_module_path = STRATEGY_MAP[strategy_name]
    
    hist_events_dict = {}
    
    all_times = None
    all_params = STRATEGY_OPT_CONFIG_MAP[strategy_name]
    fin_df_arr = []
    
    all_times, hist_events_dict = utils.generate_bar_data(tickers, start, end, bar_method='all')    
    all_times.sort()
    train_idx = int(len(all_times)*tvt_ratio[0])
    val_idx = int(len(all_times)*(tvt_ratio[0] + tvt_ratio[1]))
    train_times = all_times[:train_idx]
    val_times = all_times[train_idx:val_idx]
    all_events_consolidated = utils.create_consolidated_events_adj_ohlcv(hist_events_dict)
    for idx, params in enumerate(all_params):
        print("running for {}".format(params))
        try:
            strategy = getattr(importlib.import_module(full_module_path), strategy_name)(run_name, initial_capital, 0, tickers, **params)
            strategy.db_loc = db_loc if db_loc is not None else strategy.db_loc
            for np_dt in train_times:
                events = all_events_consolidated[all_events_consolidated['TimeStamp'] == np_dt]
                if not strategy.skip_event(events):
                    strategy.db_write_mkt(events)
            strategy.prepare_training_data()
            strategy.train_model()
            insample_scores = strategy.generate_scores(mode='insample')
            for np_dt in val_times:
                events = all_events_consolidated[all_events_consolidated['TimeStamp'] == np_dt]
                if not strategy.skip_event(events):
                    strategy.db_write_mkt(events)
                    strategy.update_indicators()
                    if np_dt != val_times[-1]:
                        strategy.generate_signal()
                        strategy.place_orders()
                        strategy.update_essential_data()
                    else:
                        strategy.square_off()
            strategy.prepare_validation_data()
            strategy.validate_model()
            outsample_scores = strategy.generate_scores(mode='outsample')
            perf_metrics = strategy.compute_perf_metrics()
            perf_metrics.update(flatten(insample_scores, reducer='dot'))
            perf_metrics.update(flatten(outsample_scores, reducer='dot'))
            perf_metrics.update(flatten(params, reducer='dot'))
            perf_df = pd.DataFrame(perf_metrics, index=[idx])
            fin_df_arr.append(perf_df)
        except Exception as e:
            print("some error, skipping: {}".format(e))
            continue
    fin_df = pd.concat(fin_df_arr, axis=0)
    table_loc = strategy.db_loc + strategy.identifier + "-optres.csv"
    fin_df.to_csv(table_loc)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse optimizer arguments')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--strategy_name', default='MULTICLASS_CLASSIFIER_REBAL')
    parser.add_argument('--tickers')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--initial_capital', default=1000000, type=float)
    parser.add_argument('--tvt_ratio', default='0.6,0.2,0.2', type=str)
    parser.add_argument('--db_loc', default=None)
    args = parser.parse_args()
    tickers = args.tickers.split(',') if args.tickers is not None else None
    tvt_ratio = [float(x) for x in args.tvt_ratio.split(',')]
    if len(tvt_ratio) != 3:
        raise Exception("tvt_ratio should be an array of size 2 for regular optimize")
    if np.sum(np.array(tvt_ratio)) != 1.0:
        raise Exception("tvt_ratio specified should sum up to 1")
    main(
        run_name=args.run_name,
        strategy_name=args.strategy_name,
        tickers=tickers,
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        tvt_ratio=tvt_ratio,
        db_loc=args.db_loc
    )