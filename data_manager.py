import os
import os.path
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Integer, Float, DateTime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


class DataBase:

    def __init__(self, db_name):

        self.db_name = db_name
        self.ticker_name = None
        self.start_date = None
        self.end_date = None
        self.data = pd.DataFrame()
        self.levels = []
        self.engine = create_engine(f'sqlite:///{self.db_name}', echo=False)
        self.variable_types = {'Open': Float,
                               'High': Float,
                               'Close': Float,
                               'Low': Float,
                               'Volume': Integer,
                               'Dividends': Float,
                               'Stock Splits': Float,
                               'Date': DateTime,
                               'Week_day': Integer,
                               'High_range': Float,
                               'Low_range': Float,
                               'Candle_type': Integer,
                               'Resistance_level': Float,
                               'Pct_to_resistance': Float,
                               'Support_level': Float,
                               'Pct_to_support': Float,
                               'Profit_day': Integer,
                               'High_range_forec': Float,
                               'Profit_day_forec': Float,
                               'High_range_hat': Float,
                               'Low_range_hat': Float,
                               'Profit_day_hat': Float}

    def exists(self):
        if os.path.exists(self.db_name):
            return True
        else:
            print(f"DB {self.db_name} doesn't exist")

    def create(self):
        if not self.exists():
            engine = self.engine
            connection = engine.connect()
            connection.close()
            print(f'DB {self.db_name} created')
        else:
            print(f"Connecting to {self.db_name}")

    def delete(self):
        if self.exists():
            os.remove(self.db_name)
            print(f'DB {self.db_name} removed!')

    def clear(self):
        if not self.exists():
            self.create()
        else:
            self.delete()
            self.create()

    def get_data(self, ticker_name, start_date, end_date=None):
        print(f'Downloading data to {self.db_name}')
        self.ticker_name = ticker_name
        self.start_date = start_date
        self.end_date = end_date
        ticker = yf.Ticker(self.ticker_name)
        if self.end_date != None:
            self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d') + timedelta(days=1)
        df = ticker.history(start=self.start_date, end=self.end_date)
        df.reset_index(inplace=True)
        self.data = df
        return self.data

    def get_all_data(self, ticker_name):
        print(f'Downloading data to {self.db_name}')
        self.ticker_name = ticker_name
        ticker = yf.Ticker(self.ticker_name)
        df = ticker.history(period="max")
        df.reset_index(inplace=True)
        self.data = df
        return self.data

    def insert_columns(self):
        for variable in self.variable_types:
            if variable not in self.data.columns:
                self.data[variable] = 0
            self.data = self.data
        return self.data

    def set_week_day(self):
        df = self.data
        df.Week_day = df.Date.apply(lambda x: x.weekday())
        self.data = df

    def set_high_range(self):
        df = self.data
        df.High_range = (df.High - df.Open) / df.Open
        self.data = df

    def set_low_range(self):
        df = self.data
        df.Low_range = (df.Low - df.Open) / df.Open
        self.data = df

    def set_candle_type(self):
        ''' Set an integer value in 'Candle_type' column depending on that day's candle OHLC features '''

        df = self.data
        for row in range(len(df)):
            if df.loc[row, 'Open'] > df.loc[row, 'Close']:
                if df.loc[row, 'Open'] < df.loc[row, 'High']:
                    if df.loc[row, 'Close'] == df.loc[row, 'Low']:
                        df.loc[row, 'Candle_type'] = 1
                    elif df.loc[row, 'Close'] > df.loc[row, 'Low']:
                        df.loc[row, 'Candle_type'] = 2
                elif df.loc[row, 'Open'] == df.loc[row, 'High']:
                    if df.loc[row, 'Close'] == df.loc[row, 'Low']:
                        df.loc[row, 'Candle_type'] = 3
                    elif df.loc[row, 'Close'] > df.loc[row, 'Low']:
                        df.loc[row, 'Candle_type'] = 4
            elif df.loc[row, 'Open'] < df.loc[row, 'Close']:
                if df.loc[row, 'Open'] > df.loc[row, 'Low']:
                    if df.loc[row, 'Close'] == df.loc[row, 'High']:
                        df.loc[row, 'Candle_type'] = 5
                    elif df.loc[row, 'Close'] < df.loc[row, 'High']:
                        df.loc[row, 'Candle_type'] = 6
                elif df.loc[row, 'Open'] == df.loc[row, 'Low']:
                    if df.loc[row, 'Close'] == df.loc[row, 'High']:
                        df.loc[row, 'Candle_type'] = 7
                    elif df.loc[row, 'Close'] < df.loc[row, 'High']:
                        df.loc[row, 'Candle_type'] = 8
            else:
                df.loc[row, 'Candle_type'] = 9
        self.data = df

    def set_profit_threshold(self, return_threshold):
        ''' Set a boolean value in 'Profit_day' column depending on if a given return_threshold met that day (0) or not (1)'''
        df = self.data
        for row in range(len(df)):
            if df.loc[row, 'High_range'] >= return_threshold:
                df.loc[row, 'Profit_day'] = 1
            else:
                df.loc[row, 'Profit_day'] = 0
        self.data = df

    def thres_probab_dist(self):
        '''Returns the probability distribution for each high_range percentage return threshold on the ticker price'''

        df = self.data
        thres_list = []
        prob_list = []
        for thres in np.arange(0.0005, 0.10, 0.0001):
            thres_list.append(thres)
            prob = df.loc[df.High_range > thres, 'High_range'].count() / df.shape[0]
            prob_list.append(prob)
        return prob_list, thres_list

    def return_threshold(self, probability, echo=False):
        """
        :param probability: float
            Probability of days that have to meet the threshold condition (0-1)
        :param echo: bool
            If True, prints a threshold explanation
        :return: float
        Returns the high_range threshold of a given probability, according to historical data"""

        prob_list, thres_list = self.thres_probab_dist()
        threshold = 0
        for element in zip(prob_list, thres_list):
            if element[0] >= probability:
                threshold = element[1]
        if echo:
            print(f'The {round(probability * 100, 2)}% of days the expected High_range return is equal or lower than {round(threshold * 100, 2)}%')

        return threshold

    def return_probability(self, threshold, echo=False):
        '''Returns the probability of a given High_range percentage return, according to historical data'''

        prob_list, thres_list = self.thres_probab_dist()
        probability = 0
        for element in zip(prob_list, thres_list):
            if element[1] <= threshold:
                probability = element[0]
        if echo:
            print(f'A high_range of {round(threshold * 100, 2)}% has a probability of appearing equal or lower than {round(probability * 100, 2)}%')

        return probability

    def plot_thres_probab_dist(self):
        '''Show a plot of probability of apearing each high_range threshold level'''

        probability, threshold = self.thres_probab_dist()
        data = pd.DataFrame(columns=['Probability %', 'Threshold %'])
        data['Probability %'] = [x * 100 for x in probability]
        data['Threshold %'] = [x * 100 for x in threshold]
        data.plot(x='Threshold %', y='Probability %', marker='.')

    def plot_cumsum_probab_dist(self):
        '''Show a plot of the accumulated probabilistic loss for each additional high_range threshold level'''

        probability, threshold = self.thres_probab_dist()
        prob_cumsum = np.cumsum(np.diff(probability))
        data = pd.DataFrame(columns=['Cumsum_prob_diff %', 'Return_threshold %'])
        data['Cumsum_prob_diff %'] = [x * 100 for x in prob_cumsum]
        data['Return_threshold %'] = [x * 100 for x in threshold][:-1]
        data.plot(x='Return_threshold %', y='Cumsum_prob_diff %', marker='.')

    def get_level_location(self):
        df = self.data

        def isSupport(i):
            support = df['Low'][i] < df['Low'][i - 1] and df['Low'][i] < df['Low'][i + 1] and df['Low'][i + 1] < df['Low'][i + 2] and df['Low'][i - 1] < df['Low'][i - 2]
            return support

        def isResistance(i):
            resistance = df['High'][i] > df['High'][i - 1] and df['High'][i] > df['High'][i + 1] and df['High'][i + 1] > df['High'][i + 2] and df['High'][i - 1] > df['High'][i - 2]
            return resistance

        def isFarFromLevel(l):
            s = np.mean(df['High'] - df['Low'])
            return np.sum([abs(l - x) < s for x in levels_location]) == 0
        levels_location = []
        for i in range(2, df.shape[0] - 2):
            if isSupport(i):
                l = df['Low'][i]
                if isFarFromLevel(l):
                    levels_location.append((i, l))
            elif isResistance(i):
                l = df['High'][i]
                if isFarFromLevel(l):
                    levels_location.append((i, l))
        levels = [element[1] for element in levels_location]
        self.levels = levels

        return levels, levels_location

    def set_levels(self, lower_resistance_pct=0.01, upper_support_pct=0.01):
        df = self.data
        levels = self.levels
        for row in range(len(df)):
            close_price = df.loc[row, 'Close']
            levels.append(close_price)
            levels = sorted(levels)
            close_index = levels.index(close_price)
            if close_index == 0:
                df.loc[row, 'Resistance_level'] = close_price * (1 - lower_resistance_pct)
                support_index = 1
                support_level = levels[support_index]
                df.loc[row, 'Support_level'] = support_level
            elif close_index == len(levels) - 1:
                resistance_index = len(levels) - 2
                resistance_level = levels[resistance_index]
                df.loc[row, 'Resistance_level'] = resistance_level
                df.loc[row, 'Support_level'] = close_price * (1 + upper_support_pct)
            else:
                resistance_index = close_index - 1
                resistance_level = levels[resistance_index]
                df.loc[row, 'Resistance_level'] = resistance_level
                support_index = close_index + 1
                support_level = levels[support_index]
                df.loc[row, 'Support_level'] = support_level

            levels.remove(close_price)
        self.data = df
        return self.data

    def set_pct_to_levels(self):
        df = self.data
        for row in range(len(df)):
            df.loc[row, 'Pct_to_support'] = (df.loc[row, 'Support_level'] - df.loc[row, 'Close']) / df.loc[row, 'Close']
            df.loc[row, 'Pct_to_resistance'] = (df.loc[row, 'Resistance_level'] - df.loc[row, 'Close']) / df.loc[
                row, 'Close']
        self.data = df
        return self.data

    def fill_dates(self, return_threshold=0.01, freq='C'):
        df = self.data
        df.set_index('Date', inplace=True)
        df = df.asfreq(freq).fillna(0)
        df.reset_index(inplace=True)
        close_price = 0
        for row in range(len(df)):
            if df.loc[row, 'Volume'] == 0:
                df.loc[row, 'Open'] = close_price
                df.loc[row, 'High'] = close_price
                df.loc[row, 'Low'] = close_price
                df.loc[row, 'Close'] = close_price
                df.loc[row, 'Dividends'] = 0
                df.loc[row, 'Stock Splits'] = 0.0
            close_price = df.iloc[row, 3]
        self.data = df
        self.set_week_day()
        self.set_high_range()
        self.set_low_range()
        self.set_candle_type()
        self.set_profit_threshold(return_threshold)
        self.get_level_location()
        self.set_levels()
        self.set_pct_to_levels()
        self.data = df
        return self.data

    def set_predictions(self):

        """Set IA predicted values for next day based on historical data"""
        #Sustituir los algoritmos por predicciones en el dia anterior al predicho
        df = self.data
        df.loc[:, ['High_range_hat']] = df.High_range.shift(-1) #Suponemos prediccion 100% acertada
        df.loc[:, ['High_range_forec']] = df.High_range_hat.shift(1)
        df.loc[:, ['High_range_error']] = df.High_range - df.High_range_forec
        df.loc[:, ['Profit_day_hat']] = df.Profit_day.shift(-1) #Suponemos prediccion 100% acertada
        df.loc[:, ['Profit_day_forec']] = df.Profit_day_hat.shift(1)
        df.loc[:, ['Profit_day_error']] = df.Profit_day - df.Profit_day_forec
        df.loc[:, ['Low_range_hat']] = df.Low_range.shift(-1) #Suponemos prediccion 100% acertada
        df.loc[:, ['Low_range_forec']] = df.Low_range_hat.shift(1)
        df.loc[:, ['Lower_day_error']] = df.Low_range - df.Low_range_forec
        df = df.dropna()

        self.data = df
        print('Generating forecast values for next day')
        print('Forecast values for next day generated!')

    def enrich_data(self, return_threshold=0.01):

        """Set new data on each data row generated on that row data"""

        print('Generating new data')
        self.insert_columns()
        self.set_week_day()
        self.set_high_range()
        self.set_low_range()
        self.set_candle_type()
        self.set_profit_threshold(return_threshold)

    def set_indicators(self):

        """Set new indicators on each data row based on historical data"""

        self.get_level_location()
        self.set_levels()
        self.set_pct_to_levels()
        self.fill_dates()
        self.set_predictions()

    def save_data(self, ticker_name, start_date, end_date=None, if_exists='replace'):
        self.data = self.get_data(ticker_name, start_date, end_date)
        if not self.data.empty:
            self.enrich_data()
            self.data.to_sql(self.ticker_name, self.engine, if_exists=if_exists, index=False, dtype=self.variable_types)
            self.load_data(self.ticker_name)
            self.set_indicators()
            self.data.to_sql(self.ticker_name, self.engine, if_exists=if_exists, index=False, dtype=self.variable_types)
            print(f'Data saved in {self.db_name}/{self.ticker_name}')
            return self.data
        else:
            print("There aren't data loaded")

    def save_all_data(self, ticker_name, if_exists='replace'):
        self.data = self.get_all_data(ticker_name)
        if not self.data.empty:
            self.enrich_data()
            self.data.to_sql(self.ticker_name, self.engine, if_exists=if_exists, index=False, dtype=self.variable_types)
            self.load_data(self.ticker_name)
            self.set_indicators()
            self.data.to_sql(self.ticker_name, self.engine, if_exists=if_exists, index=False, dtype=self.variable_types)
            print(f'Data saved in {self.db_name}/{self.ticker_name}')
            return self.data
        else:
            print("There aren't data loaded")

    def load_data(self, ticker_name, chop_start=None, chop_end=None):
        self.ticker_name = ticker_name
        if not self.exists():
            pass
        else:
            df = pd.read_sql_table(self.ticker_name, con=self.engine)
            df.set_index('Date', inplace=True)
            df = df[chop_start:chop_end]
            df.reset_index(inplace=True)
            self.data = df
            print(f'Data load from {self.db_name}/{self.ticker_name}')
            return self.data

    def last_date(self, ticker_name):
        self.ticker_name = ticker_name
        if not self.exists():
            pass
        else:
            last_date = pd.read_sql_table(self.ticker_name, con=self.engine).Date.iloc[-1]
            return last_date

    def is_updated(self, ticker_name):
        self.ticker_name = ticker_name
        if not self.exists():
            pass
        else:
            last_day = (self.last_date(self.ticker_name)).strftime('%Y-%m-%d')
            today = datetime.today().strftime('%Y-%m-%d')
            if last_day != today:
                print(f'Updating data stored in {self.db_name}/{self.ticker_name} from {last_day} to {today}')
                return False
            else:
                print('DB up to date!')
                return True

    def update_data(self, ticker_name):
        self.ticker_name = ticker_name
        if not self.exists():
            pass
        else:
            if self.is_updated(self.ticker_name):
                pass
            else:
                last_update = self.last_date(self.ticker_name)
                last_day = last_update.to_pydatetime().date()
                last_weekday = last_update.to_pydatetime().weekday()
                today = datetime.today().date()

                if last_weekday == 4:
                    if today - timedelta(1) == last_day:
                        print('Data up to date!')
                    elif today - timedelta(2) == last_day:
                        print('Data up to date!')
                    else:
                        start_date = last_update + timedelta(days=3)
                        self.save_data(self.ticker_name, start_date=start_date, if_exists='append')
                        print('Data updated!')
                else:
                    start_date = last_update + timedelta(days=1)
                    self.save_data(self.ticker_name, start_date=start_date, if_exists='append')
                    print('Data updated!')



#db = DataBase('OHLC.db')
#df = db.get_data('GME', start_date='2021-08-25')
#df.set_index('Date', inplace=True)
#db.save_all_data('GME')
#db.update_data('GME')
#df = db.load_data('GME')
#print(df.tail(20))
#df = db.get_data('GME', start_date='2022-01-10', end_date='2022-08-19')
#df = db.get_data('GME', start_date='2022-08-22', end_date='2022-08-27')
#print(df)
#print(df.loc[df.Volume == 0])
#print(df)