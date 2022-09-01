from data_manager import DataBase
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class Bcktest:

    def __init__(self, db_name, ticker_name, start_date=None, end_date=None):

        self.db_name = db_name
        self.ticker_name = ticker_name
        self.threshold = 0.01
        self.comission = 1
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = 1000
        self.cash = self.initial_cash
        self.equity = self.initial_cash - self.cash
        self.stock_units = 0
        self.buy_price = 0
        self.sell_price = 0
        self.day_profit = 0
        self.day_return = 0
        self.profit_cumsum = 0
        self.return_cumsum = 0
        self.streak_list = []
        self.mean_streak = np.mean(self.streak_list)
        #self.max_streak = np.amax(np.array(self.streak_list))
        #self.data = pd.DataFrame()

    def set_strategy(self):
        # Loads data
        db = DataBase(self.db_name)
        df = db.load_data(self.ticker_name, chop_start=self.start_date, chop_end=self.end_date)
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
        df.loc[date, 'Profit_cumsum'] = self.profit_cumsum
        df.loc[date, 'Return_cumsum'] = self.return_cumsum
        df = df.fillna(0)
        self.data = df

        return self.data

    def sell(self, date):
        df = self.data
        self.sell_price = df.loc[date, 'Open'] * (1 + self.threshold)
        self.day_profit = ((self.sell_price * self.stock_units) - (self.buy_price * self.stock_units)) - (self.comission * 2)
        self.cash = self.cash + self.equity + self.day_profit
        self.day_return = self.day_profit / self.equity ######
        self.profit_cumsum = self.profit_cumsum + self.day_profit
        self.return_cumsum = ((self.cash + self.equity) - self.initial_cash) / self.initial_cash
        self.stock_units = self.stock_units - self.stock_units
        self.equity = self.equity - self.equity
        df.loc[date, 'Sell_price'] = self.sell_price
        df.loc[date, 'Cash'] = self.cash
        df.loc[date, 'Equity'] = self.equity
        df.loc[date, 'Day_profit'] = self.day_profit
        df.loc[date, 'Day_return'] = self.day_return
        df.loc[date, 'Profit_cumsum'] = self.profit_cumsum
        df.loc[date, 'Return_cumsum'] = self.return_cumsum
        df.loc[date, 'Stock_units'] = self.stock_units
        df = df.fillna(0)
        self.data = df
        self.day_profit = 0
        self.day_return = 0

        return self.data

    def hold(self, date):
        df = self.data
        self.day_profit = 0
        self.day_return = 0
        df.loc[date, 'Buy_price'] = 0
        df.loc[date, 'Sell_price'] = 0
        df.loc[date, 'Cash'] = self.cash
        df.loc[date, 'Equity'] = self.equity
        df.loc[date, 'Stock_units'] = self.stock_units
        df.loc[date, 'Day_profit'] = 0
        df.loc[date, 'Day_return'] = 0
        df.loc[date, 'Profit_cumsum'] = self.profit_cumsum
        df.loc[date, 'Return_cumsum'] = self.return_cumsum
        df = df.fillna(0)
        self.data = df


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
        i = 0
        for date in self.data.index:
            i = i + 1
            if self.data.loc[date, 'Buy']:
                if self.stock_units == 0:
                    if self.cash >= self.data.loc[date, 'Open']:
                        self.buy(date)
                        # print(date, self.stock_units)
            if self.data.loc[date, 'Sell']:
                if self.stock_units > 0:
                    self.sell(date)

            if not self.data.loc[date, 'Buy'] and not self.data.loc[date, 'Sell']:
                self.hold(date)

            #Streak
            if self.data.loc[date, 'Profit_day'] != self.data.loc[date, 'Profit_day_forec']:
                self.streak_list.append(i)
                i = 0

        # reset index
        # enviar a la base de datos
        return self.data

    def confusion_matrix(self, y, y_forec):
        df = self.data
        cm = confusion_matrix(df[y], df[y_forec])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def aur_roc(self, y, y_forec):
        df = self.data
        sns.set()
        fpr, tpr, thresholds = roc_curve(df[y], df[y_forec])
        plt.plot(fpr, tpr)
        plt.plot(fpr, fpr, linestyle='--', color='k')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        AUROC = np.round(roc_auc_score(df[y], df[y_forec]), 2)
        plt.title(f'Logistic Regression Model ROC curve; AUROC: {AUROC}')
        plt.show()

    def precision_recall(self, y, y_forec):
        df = self.data
        average_precision = average_precision_score(df[y], df[y_forec])
        precision, recall, thresholds = precision_recall_curve(df[y], df[y_forec])
        plt.plot(recall, precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title(f'Precision Recall Curve. AUPRC: {average_precision}')
        plt.show()

    def evaluate(self, y, y_forec):
        df = self.data
        print(f"Precision Score: {precision_score(df[y], df[y_forec])}")
        print(f"Recall Score:    {recall_score(df[y], df[y_forec])}")
        print(f"F1 Score:        {f1_score(df[y], df[y_forec])}")
        #print(f"Average streak:  {self.mean_streak}")
        #print(f"Maximum streak:  {self.max_streak}")

    def plot(self):
        df = self.data
        df.reset_index(inplace=True)
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        sns.lineplot(x="Date", y="Open", color='g', data=df, ax=ax[0][0])

        ax[0][0].tick_params(labelrotation=15)
        sns.lineplot(x="Date", y="High_range", color='b', data=df, ax=ax[0][1])

        ax[0][1].tick_params(labelrotation=15)
        sns.lineplot(x="Date", y="Day_profit", color='r', data=df, ax=ax[1][0])

        ax[1][0].tick_params(labelrotation = 15)
        sns.lineplot(x="Date", y="Day_return", color ='y', data=df, ax=ax[1][1])

        ax[1][1].tick_params(labelrotation=15)
        fig.tight_layout(pad=1.25)
        plt.show()


bt = Bcktest('OHLC.db', 'GME', '2020-01-02')
bt.run()
#bt.evaluate('Profit_day', 'Profit_day_forec')
bt.plot()
#print(df)


#bt = Backtest(df, SmaCross, cash=10000, commission=.005, exclusive_orders=True)

#output = bt.run()
#print(output)
#bt.start()
