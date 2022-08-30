from data_manager import DataBase
import pandas as pd
import numpy as np


class Bcktest:

    def __init__(self, db_name, ticker_name, start_date=None, end_date=None):

        self.db_name = db_name
        self.ticker_name = ticker_name
        self.threshold = 0.01
        self.comission = 1
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = 10000
        self.cash = 10000
        self.equity = 0
        self.stock_units = 0
        self.buy_price = 0
        self.sell_price = 0
        self.day_profit = 0
        self.day_return = 0
        self.profit_cumsum = 0
        self.data = pd.DataFrame()

    def set_strategy(self):
        # Loads data
        db = DataBase(self.db_name)
        df = db.load_data(self.ticker_name, chop_start=self.start_date,
                          chop_end=self.end_date)  # .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.set_index('Date', inplace=True)

        # Set 1 if conditions have meet during session
        df['Buy'] = df.apply(lambda row: 1 if row['Profit_day_forec'] == 1 else 0, axis=1).to_list()
        # df['Buy'] = df['Buy'].shift(1, fill_value=0)
        df['Sell'] = df.apply(lambda row: 1 if row['High_range_forec'] >= self.threshold else 0, axis=1).to_list()
        self.data = df

        return self.data

    def buy(self, date):
        df = self.data
        self.buy_price = df.loc[date, 'Open']
        self.stock_units = int(str(self.cash / self.buy_price).split('.')[0])
        self.equity = self.stock_units * self.buy_price
        self.cash = self.cash - self.equity
        df.loc[date, 'Buy_price'] = self.buy_price
        df.loc[date, 'Stock_units'] = self.stock_units
        df.loc[date, 'Equity'] = self.equity
        df.loc[date, 'Cash'] = self.cash
        df = df.fillna(0)
        self.data = df

        return self.data

    def sell(self, date):
        df = self.data
        self.sell_price = df.loc[date, 'Open'] * (1 + self.threshold)
        self.day_profit = ((self.sell_price * self.stock_units) - (self.buy_price * self.stock_units)) - (
                    self.comission * 2)
        self.cash = self.cash + self.equity + self.day_profit
        self.day_return = self.day_profit / self.equity
        self.profit_cumsum = self.profit_cumsum + self.day_profit

        self.equity = 0
        df.loc[date, 'Sell_price'] = self.sell_price
        df.loc[date, 'Cash'] = self.cash
        df.loc[date, 'Equity'] = self.equity
        df.loc[date, 'Day_profit'] = self.day_profit
        df.loc[date, 'Day_return'] = self.day_return
        df.loc[date, 'Profit_cumsum'] = self.profit_cumsum
        df.loc[date, 'Return_cumsum'] = (self.cash - self.initial_cash) / self.initial_cash
        df.loc[date, 'Stock_units'] = self.stock_units
        df = df.fillna(0)
        self.data = df
        self.day_profit = 0
        self.day_return = 0
        self.stock_units = 0

        return self.data

    # def run(self):
    # Asignamos los valores 1 para Buy y Sell según si se han ejecutado las condiciones de set_strategy()
    # self.set_strategy()
    # Iteramos para cada dia buy() y sell() en función de si son True
    # for date in self.data.index:
    #    if self.data.loc[date, 'Buy']:
    #      if self.stock_units == 0:
    #        if self.cash >= self.data.loc[date, 'Open']:
    #            self.buy(date)
    #    if self.data.loc[date, 'Sell']:
    #        if self.stock_units > 0:
    #            self.sell(date)
    # reset index
    # enviar a la base de datos
    # return self.data

    def run(self):
        # Asignamos los valores 1 para Buy y Sell según si se han ejecutado las condiciones de set_strategy()
        self.set_strategy()
        # Iteramos para cada dia buy() y sell() en función de si son True
        for date in self.data.index:
            if self.data.loc[date, 'Buy']:
                if self.stock_units == 0:
                    if self.cash >= self.data.loc[date, 'Open']:
                        self.buy(date)
                        # print(date, self.stock_units)
            if self.data.loc[date, 'Sell']:
                if self.stock_units > 0:
                    self.sell(date)
        # reset index
        # enviar a la base de datos
        return self.data

bt = Bcktest('OHLC.db', 'GME', '2020-01-02')
df = bt.run()
print(df)


#bt = Backtest(df, SmaCross, cash=10000, commission=.005, exclusive_orders=True)

#output = bt.run()
#print(output)
#bt.start()
