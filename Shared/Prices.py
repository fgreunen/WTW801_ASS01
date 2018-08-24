import quandl
import os
import pandas as pd
import datetime
import numpy as np

class Prices:
    RISK_FREE_RATE = 0.0
    TRADING_DAYS_PER_YEAR = 252.0
    def __init__(self, tickername, start = '2010-01-01', end = '2017-12-31', datadir = './Data/', apikey = 'UDGVtB83rhcmoyJXMrTG'):
        self.apikey = apikey
        self.start = datetime.datetime.strptime(start, "%Y-%m-%d")
        self.end = datetime.datetime.strptime(end, "%Y-%m-%d")
        self.tickername = tickername
        self.filename = os.path.join(datadir, 
            self.tickername + '-' + self.start.strftime("%Y_%m_%d") + '-' + self.end.strftime("%Y_%m_%d") + '.pkl')
        self._populatePrices()
    
    def getPricesResampledMonthly(self):
        return self.data.resample('M').agg(lambda x: x[-1])

    def _populatePrices(self):
        self.data = None
        if (not os.path.isfile(self.filename)):
            newData = quandl.get_table(
                'WIKI/PRICES', 
                qopts = { 'columns': ['date', 'adj_close'] }, 
                ticker = [self.tickername], 
                date = { 
                    'gte': self.start.strftime("%Y-%m-%d"), 
                    'lte': self.end.strftime("%Y-%m-%d") 
                },
                api_key=self.apikey)
            newData.to_pickle(self.filename)
        data = pd.read_pickle(self.filename)
        data = data.iloc[::-1] # reverse
        data['pct_change_1D'] = data.adj_close.pct_change(1)
        data['pct_change_1M'] = data.adj_close.pct_change(21)
        data = data.dropna()
        data['date'] = pd.DatetimeIndex(pd.to_datetime(data['date'], format='%Y-%m-%d'))
        data['excess_daily_ret'] = data['pct_change_1D'] - self.RISK_FREE_RATE/self.TRADING_DAYS_PER_YEAR
        data = data.set_index('date')

        rolled = data.rolling(int(self.TRADING_DAYS_PER_YEAR))
        data['rolling_annual_sharpe_ratio'] = np.sqrt(self.TRADING_DAYS_PER_YEAR) * rolled.mean().pct_change_1D / rolled.std().pct_change_1D
        data['rolling_annual_volatility'] = rolled.std().pct_change_1D
        data = data.fillna(value = 0)

        self.data = data
        self.sharpeRatio = np.sqrt(self.TRADING_DAYS_PER_YEAR) * data['excess_daily_ret'].mean() / data['excess_daily_ret'].std()
        self.distributionOfDailyReturns = (data.pct_change_1D.mean(), data.pct_change_1D.std())


    def getPrices(self):
        return self.data