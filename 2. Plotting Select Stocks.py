import matplotlib.pyplot as plt
import matplotlib as mpl
from Shared import Prices
import pandas as pd

# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
# https://www.quantconnect.com/tutorials/introduction-to-financial-python/pandas-resampling-and-dataframe

if __name__ == '__main__':
    prices = Prices.Prices('ABC')
    print(prices.distributionOfDailyReturns)
    print(prices.sharpeRatio)
    
    prices = prices.getPrices()
    prices.plot(y='rolling_annual_sharpe_ratio')
    prices.plot(y='rolling_annual_volatility')
    prices.plot(y='adj_close')
    # prices.plot(y='pct_change_1M')
    # prices.plot(y='pct_change_1D')
    plt.show()