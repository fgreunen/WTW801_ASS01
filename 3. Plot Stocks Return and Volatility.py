import matplotlib.pyplot as plt
import matplotlib as mpl
from Shared import Prices, Universe
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# https://blog.quantopian.com/markowitz-portfolio-optimization-2/
# http://dacatay.com/data-science/portfolio-optimization-python/

USE_DAILY_RETURNS_FOR_EXPECTED_RETURN = True

if __name__ == '__main__':
    localUniverse = sorted(Universe.Universe().getLocalUniverse())[:5]
    # localUniverse = [
    #     'AAPL',
    #     'MSFT',
    #     'GOOGL',
    #     'INTC',
    #     'AMZN',
    #     'FB',
    #     'BABA',
    #     'JNJ',
    #     'JPM',
    #     'XOM',
    #     'AIG',
    #     'PG',
    #     'MO',
    #     'TWX',
    #     'MRK',
    #     'HPQ',
    # ]
    print(localUniverse)
    stockPrices = [Prices.Prices(x) for x in localUniverse]
    stockPrices = [x for x in stockPrices if x.hasEnoughData()]

    def annualize(value, periods = 12):
        return 100 * np.sqrt(periods) * value

    periodicReturns = [x.getDataForCovarianceCalculation() for x in stockPrices]
    if (USE_DAILY_RETURNS_FOR_EXPECTED_RETURN):
        expectedReturns = annualize(np.array([x.mean() for x in periodicReturns])) # E(r)
    else:
        expectedReturns = [x.getAnnualMeanReturnAndVolatility()[0] for x in stockPrices] # E(r)

    p = np.asmatrix(expectedReturns)
    C = np.asmatrix(np.cov(periodicReturns)) # Covariance matrix

    def random_portfolio():
        n = len(periodicReturns)
        weights = np.random.rand(n)
        weights = weights / sum(weights)
        w = np.asmatrix(weights)
        mu = w * p.T
        sigma = np.sqrt(w * C * w.T)
        return float(mu[0][0]), annualize(float(sigma[0,0]))

    n_portfolios = 500
    means, stds = np.column_stack([
        random_portfolio() 
        for _ in range(n_portfolios)
    ])

    portfolios = []
    for x in range(len(means)):
        portfolios.append((means[x], stds[x]))

    portfolios = [x for x in portfolios if x[0] > 3]
    portfoliosByReturn = sorted(portfolios, key = lambda tup: tup[0], reverse = True)
    portfoliosByVolatility = sorted(portfolios, key = lambda tup: tup[1])

    plt.plot([x[1] for x in portfolios], [x[0] for x in portfolios], 'o', markersize=3)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    plt.scatter(x=portfoliosByReturn[0][1], y=portfoliosByReturn[0][0], c='red', marker='o', s=50)
    plt.scatter(x=portfoliosByVolatility[0][1], y=portfoliosByVolatility[0][0], c='green', marker='o', s=50)
    plt.show()