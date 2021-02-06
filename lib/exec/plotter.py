# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 07:22:53 2020

@author: vivin
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def _load_and_fill(db_loc, table_name, columns, norm):
    df = pd.read_csv('{}{}.csv'.format(db_loc, table_name))
    df.loc[:, 'TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df.set_index('TimeStamp', inplace=True)
    df = df[columns]
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    if norm:
        df_norm = pd.concat([df.iloc[[0], :]] * len(df), axis=0)
        df_norm.set_index(df.index.values, inplace=True)
        df = df/df_norm
    return df

def main(
        tables=['mom-rebal-algo', 'multiclass-knn-d-algo', 'multiclass-gnb-algo'], 
        columns=['Eq Curve Realized'],
        anc_table='mom-rebal-algo', 
        anc_cols=['INFRABEES.NS Price', 'HDFCMFGETF.NS Price'],
        norm=True,
        db_loc=None
):
    db_loc = db_loc if db_loc is not None else "C:\\Users\\vivin\\Documents\\QuantInsti\\project_data\\"
    df_arr = []
    for table_name in tables:
        df = _load_and_fill(db_loc, table_name, columns, norm)
        new_col_names = ['{}.{}'.format(x, table_name) for x in columns]
        df.rename(columns=dict(zip(columns, new_col_names)), inplace=True)
        df_arr.append(df)

    df_anc = _load_and_fill(db_loc, anc_table, anc_cols, norm)
    df_arr.append(df_anc)
        
    df_fin = pd.concat(df_arr, axis=1)
    df_fin.plot()
    plt.xlabel("Time")
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse optimizer arguments')
    parser.add_argument('--tables', default='mom-rebal-algo,multiclass-knn-d-algo,multiclass-gnb-algo')
    parser.add_argument('--columns', default='Eq Curve Realized')
    parser.add_argument('--anc_table', default='mom-rebal-algo')
    parser.add_argument('--anc_cols', default='INFRABEES.NS Price,HDFCMFGETF.NS Price')
    parser.add_argument('--db_loc', default=None)
    args = parser.parse_args()
    tables = [x for x in args.tables.split(',')]
    columns = [x for x in args.columns.split(',')]
    anc_cols = [x for x in args.anc_cols.split(',')]

    main(
        tables=tables,
        columns=columns,
        anc_table=args.anc_table,
        anc_cols=anc_cols,
        db_loc=args.db_loc
    )