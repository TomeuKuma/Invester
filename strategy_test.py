from data_manager import DataBase
import pandas as pd
import numpy as np

class Bcktest:

    def __init__(self, db_name, ticker_name):

        self.db_name = db_name
        self.ticker_name = ticker_name
        self.threshold = 0.01
        self.commission = 0.0005
        self.start_date = None
        self.end_date = None
        self.initial_cash = 10000
        self.cash = 10000
        self.equity = 0
        self.buy_price = 0
        self.sell_price = 0
        self.day_profit = 0
        self.day_return = 0
        self.stock_units = 0
        self.data = pd.DataFrame()

    def set_strategy(self):
        db = DataBase(self.db_name)
        df = db.load_for_bt(self.ticker_name, self.start_date,)  # .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Buy'] = df.apply(lambda row: 1 if row['Profit_day_hat'] == 1 else 0, axis=1).to_list()
        df['Buy'] = df['Buy'].shift(1, fill_value=0)
        df['Sell'] = df.apply(lambda row: 1 if row['High_range_hat'] >= self.threshold else 0, axis=1).to_list()
        self.data = df

    def buy(self, date):
        self.data['Buy_price'] = 0
        self.buy_price = self.data.loc[date, 'Open']
        self.stock_units = self.cash / self.buy_price
        self.equity = self.stock_units * self.buy_price
        self.cash = self.cash - self.equity
        df = self.data
        df.loc[date, 'Buy_price'] = self.buy_price
        df.loc[date, 'Stock_units'] = self.stock_units
        df.loc[date, 'Equity'] = self.equity
        df.loc[date, 'Cash'] = self.cash
        self.data = df

        return self.data

    def sell(self, date):
        df = self.data
        self.sell_price = df.loc[date, 'Open'] * (1 + self.threshold)
        self.cash = self.cash + (self.sell_price * self.stock_units)
        self.day_profit = (((self.sell_price * self.stock_units) * (1-(self.commission*2))) - self.equity)
        self.day_return = self.day_profit / self.equity
        self.stock_units = 0
        self.equity = 0
        df.loc[date, 'Sell_price'] = self.sell_price
        df.loc[date, 'Cash'] = self.cash
        df.loc[date, 'Day_profit'] = self.day_profit
        df.loc[date, 'Day_return'] = self.day_return
        df.loc[date, 'Stock_units'] = self.stock_units
        df.loc[date, 'Equity'] = self.equity
        self.data = df
        self.day_profit = 0
        self.day_return = 0

        return self.data

    def run(self):
        for date in self.data.index:
            if self.data.loc[date, 'Buy']:
                if self.cash >= self.data.loc[date, 'Open']:
                    self.buy(date)
            if self.data.loc[date, 'Sell']:
                if self.stock_units > 0:
                    self.sell(date)

        return self.data


        #Por cada una de las filas del chop
        # Si se ha cuplido Profit day:
            # Si cash <= Open
                # comprar()
        # Si se ha cumplido High_range >= threshold:
            # Si stock_units >= 0:
                #vender()


bt = Bcktest('OHLC.db', 'GME')
df = bt.run()
print(df)


#bt = Backtest(df, SmaCross, cash=10000, commission=.005, exclusive_orders=True)

#output = bt.run()
#print(output)
#bt.start()
