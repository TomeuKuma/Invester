import os
import os.path
import itertools
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time
import pandas as pd
from data_manager import DataBase
import matplotlib.pyplot as plt
import random
import seaborn as sns
from collections.abc import Iterable
from pycaret.time_series import TSForecastingExperiment
pd.set_option('display.max_columns', None)


db_name = 'OHLC.db'
ticker_name = 'GME'
#start_date = None
#end_date = None


inicio = time.time()
db = DataBase(db_name)
#db.save_all_data(ticker_name)
df = db.load_data(ticker_name)
df.set_index('Date', inplace=True)
#print(df)

#data = df['2020-01-01':'2022-08-23']
#new_data = df['2022-08-24':]
y = 'High_range'
session_id = 123
system_log = False
verbose = False
forecast_horizon = 3
fold = 1
model_name = 'forecast_model'


#df['Date'] = pd.to_datetime(df['Date'])
#df = df.set_index('Date').asfreq('C').fillna(0).drop(columns='index')
#print(df)

ts_model = TSForecastingExperiment()
ts_model.setup(data=df, target=y, fh=forecast_horizon, system_log=system_log, verbose=verbose, fold=fold, session_id=session_id)
#best = ts_model.compare_models()
best = ts_model.create_model('auto_arima')
ts_model.plot_model(best)
#tuned_best = ts_model.tune_model(best)
prediction = ts_model.predict_model(best)
#ts_model.plot_model(prediction)
print(prediction)
for date in prediction.index:
    df.reset_index(inplace=True)
    print(df.loc[df['Date'] == datetime.strptime(str(date), '%Y-%m-%d'), 'High_range'].values)
    prediction.loc[date, 'y'] = df.loc[df['Date'] == datetime.strptime(str(date), '%Y-%m-%d'), 'High_range'].values
    prediction.loc[date, 'error'] = prediction.loc[date, 'y'] - prediction.loc[date, 'y_pred']
    prediction.loc[date, 'pct_deviation'] = (prediction.loc[date, 'y_pred'] - prediction.loc[date, 'y'] / prediction.loc[date, 'y']) * 100
print(prediction)
    #print(df.loc[df.Date == datetime.strptime(str(date), '%Y-%m-%d'), 'High_range'])
    #prediction.loc[date, 'y'] = df.loc[date, 'High_range']
    #prediction.loc[date, 'error'] = df.loc[date, 'High_range']

ts_model = TSForecastingExperiment()
ts_model.setup(data=df, target='Profit_day', fh=forecast_horizon, system_log=system_log, verbose=True, fold=fold, session_id=session_id)
best = ts_model.compare_models()
#best = ts_model.create_model('auto_arima')
#ts_model.plot_model(best)
#tuned_best = ts_model.tune_model(best)
prediction = ts_model.predict_model(best)
#ts_model.plot_model(prediction)
print(prediction)
for date in prediction.index:
    df.reset_index(inplace=True)
    print(df.loc[df['Date'] == datetime.strptime(str(date), '%Y-%m-%d'), 'High_range'].values)
    prediction.loc[date, 'y'] = df.loc[df['Date'] == datetime.strptime(str(date), '%Y-%m-%d'), 'High_range'].values
    prediction.loc[date, 'error'] = prediction.loc[date, 'y'] - prediction.loc[date, 'y_pred']
    prediction.loc[date, 'pct_deviation'] = (prediction.loc[date, 'y_pred'] - prediction.loc[date, 'y'] / prediction.loc[date, 'y']) * 100
print(prediction)

#print(prediction)

fin = time.time()
print('Ha tardado en ejecutarse:', fin-inicio)

#best = create_model('theta')
#tuned_best = tune_model(best)
#plot_model(best)
#save_model(best, model_name)
#load_model = load_model(moadel_name)