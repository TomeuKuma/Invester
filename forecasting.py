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
from omegaconf import OmegaConf
config = OmegaConf.load('config.yml')

#Data
db_name = config.database.name
ticker_name = config.database.ticker
y = 'High_range'
y_hat = 'High_range_hat'
index = 'Date'

#Parameters
session_id = 123
system_log = False
verbose = False
train_forecast_horizon = 1
pred_forecast_horizon = 1
fold = 5
model_name = 'forecast_model'


def search_model(df, y):
    ts_model = TSForecastingExperiment()
    ts_model.setup(data=df, target=y, fh=train_forecast_horizon, system_log=system_log, verbose=verbose, fold=fold, session_id=session_id)
    best = ts_model.compare_models(n_select = 3)
    #tuned_best = ts_model.tune_model(best)
    return best


# Se le tiene que pasar un df con size [:dia actual]
# df.loc[dia_actual: 'y_hat'] = return del predict
def predict(df, y, model, class_threshold=None):
    ts_model = TSForecastingExperiment()
    ts_model.setup(data=df, target=y, fh=train_forecast_horizon, system_log=system_log, verbose=verbose, fold=fold, session_id=session_id, enforce_exogenous=False)
    print('Extra trees model initiated. Training model...')
    best_model = ts_model.create_model(model)
    #pred_holdout = ts_model.predict_model(best_model)
    print('Model prediction...')
    prediction = ts_model.predict_model(ts_model.finalize_model(best_model), fh=pred_forecast_horizon)
    prediction = prediction.values[0][0]

    if class_threshold:
        if prediction >= class_threshold:
            return 1
        else:
            return 0
    else:
            return prediction


start_time = time.time()
print('Intiating process')
db = DataBase('OHLC.db')
#db.save_all_data('GME')
data = db.load_data('GME')
df = data.loc[:, [index, y, y_hat]]
#print(df)


df.set_index('Date', inplace=True)
first_date = df.index[0]
last_date = df.index[-1]
start_cutoff = datetime.strptime('2015-01-02', '%Y-%m-%d')
end_cutoff = datetime.strptime('2022-08-22', '%Y-%m-%d')
df[y_hat] = 0

while True:
    if end_cutoff != last_date:
        data = df.loc[start_cutoff:end_cutoff, y]
        prediction = predict(data, y, model='et_cds_dt', class_threshold=0.01)
        df.loc[end_cutoff, y_hat] = prediction
        #print(end_cutoff, prediction)
        if end_cutoff.weekday() in [0, 1, 2, 3, 6]:
            end_cutoff = end_cutoff + timedelta(1)
        elif end_cutoff.weekday() == 4:
            end_cutoff = end_cutoff + timedelta(3)
        elif end_cutoff.weekday() == 5:
            end_cutoff = end_cutoff + timedelta(2)
    else:
        data = df.loc[start_cutoff:end_cutoff, y]
        # print(df)
        prediction = predict(data, y, model='et_cds_dt', class_threshold=0.01)
        df.loc[end_cutoff, y_hat] = prediction
        #print(end_cutoff, prediction)
        break

print("--- %s minutes ---" % ((time.time() - start_time)/60))
print(df)