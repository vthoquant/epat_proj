@echo off
:: set TICKERS=INFRABEES.NS,HDFCMFGETF.NS
set "TICKERS=^^NSEI,HDFCMFGETF.NS"
set "BASE_DIR=C:\\Users\\vivin\\Documents\\QuantInsti\\project_data\\"
set "VANILLA_FOLDER=vanilla\\"

cd "C:\Users\vivin\Downloads\auto_X-master\epat_proj\"

echo =======================================================
echo Optimize momentum rebal with vanilla historical data
echo =======================================================
:: python -m lib.exec.optimize --tickers=%TICKERS% --run_name=mom-rebal --db_loc=%BASE_DIR%%VANILLA_FOLDER%
echo ===============================================================================================
echo Optimize all multiclass reallocation algos with vanilla historical data
echo ===============================================================================================
python -m lib.exec.optimize_ml --tickers=%TICKERS% --run_name=multiclass-all --db_loc=%BASE_DIR%%VANILLA_FOLDER%