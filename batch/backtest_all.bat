@echo off
:: set TICKERS=INFRABEES.NS,HDFCMFGETF.NS
set "TICKERS=^^NSEI,HDFCMFGETF.NS"
set "BASE_DIR=C:\\Users\\vivin\\Documents\\QuantInsti\\project_data\\"
set "VANILLA_FOLDER=vanilla\\"

cd "C:\Users\vivin\Downloads\auto_X-master\epat_proj\"

echo =======================================================
echo Backtesting momentum rebal with vanilla historical data
echo =======================================================
python -m lib.exec.backtest --tickers=%TICKERS% --run_name=mom-rebal --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo ===============================================================================================
echo Backtesting all multiclass reallocation algos with vanilla historical data
echo ===============================================================================================
python -m lib.exec.backtest_ml --tickers=%TICKERS% --run_name=multiclass-dectree --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo Completed decisiontree
python -m lib.exec.backtest_ml --tickers=%TICKERS% --run_name=multiclass-svc --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo Completed SVC
python -m lib.exec.backtest_ml --tickers=%TICKERS% --run_name=multiclass-knn-d --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo Completed KNN with distance weights
python -m lib.exec.backtest_ml --tickers=%TICKERS% --run_name=multiclass-knn-u --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo Completed KNN with uniform weights
python -m lib.exec.backtest_ml --tickers=%TICKERS% --run_name=multiclass-xgb --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo Completed XGB
