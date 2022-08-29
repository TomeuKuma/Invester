from data_manager import DataBase
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
from pathlib import Path
config = OmegaConf.load('config.yml')
#print(OmegaConf.to_yaml(config))

db_name = config.database.name
ticker_name = config.database.ticker
start_date = config.database.start_date
end_date = config.database.end_date

def main():
    #pass
    db = DataBase(db_name)
    df = db.load_data('AAPL', '2022-01-01')
    df.set_index('Date', inplace=True)
    print(df)
    df.Profit_day_hat = df.Profit_day_hat.shift(-1, fill_value=None)
    print(df)
    #db.data = df[None:None]
    #db.data = df.loc[:, ['2022-01-01']]
    #print(df)
    #print(db.return_probability(0.01))
    #db.set_predictions()
    #df = db.load_data(ticker_name)
    #print(df)
    #print(db.return_probability(0.01))
    #print(db.return_threshold(0.80835734870317))

    #print(db.load_data(ticker_name))
    #db.delete()
    #df = db.save_data(ticker_name, start_date, end_date)
    #print(df)
    #db.update_data(ticker_name)
    #print(db.load_data(ticker_name))

#db.delete()
#db.get_data('GME', start_date='2022-07-01', end_date='2022-07-02')
#df = db.save_data()
#db.last_date('GME')
#db.update_data('GME')
#db.load_data('GME')

if __name__ == '__main__':
    main()



