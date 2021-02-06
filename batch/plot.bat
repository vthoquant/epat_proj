@echo off
set TICKERS=INFRABEES.NS,HDFCMFGETF.NS
::set "TICKERS=^^NSEI,HDFCMFGETF.NS"
set "BASE_DIR=C:\\Users\\vivin\\Documents\\QuantInsti\\project_data\\infra_non_normalized\\"
set "ALGO_TABLES=mom-rebal-algo,multiclass-knn-d-algo,multiclass-knn-u-algo,multiclass-svc-algo,multiclass-xgb-algo,multiclass-dectree-algo"
set "ANC_COLS="INFRABEES.NS Price,HDFCMFGETF.NS Price"
::set "ANC_COLS="^^NSEI Price,HDFCMFGETF.NS Price"
set "VANILLA_FOLDER=vanilla\\"
set "MIRROR_FOLDER=mirror\\"
set "TRIVIAL_FOLDER=trivial\\"
set "SHUFFLE_FOLDER_1=shuffle_5_101\\"
set "SHUFFLE_FOLDER_2=shuffle_5_505\\"
set "SHUFFLE_FOLDER_3=shuffle_20_1001\\"
set "SHUFFLE_FOLDER_4=shuffle_20_5005\\"

cd "C:\Users\vivin\Spyder projects\auto_X\lib\examples"

echo =======================================================
echo Plot algo performances against each other and also against the underlying assets 
echo =======================================================
python plotter.py --tables=%ALGO_TABLES% --db_loc=%BASE_DIR%%VANILLA_FOLDER% --anc_cols=%ANC_COLS%