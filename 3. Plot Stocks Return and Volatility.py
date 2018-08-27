import matplotlib.pyplot as plt
import matplotlib as mpl
from Shared import Prices, Universe
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# https://blog.quantopian.com/markowitz-portfolio-optimization-2/
# http://dacatay.com/data-science/portfolio-optimization-python/

if __name__ == '__main__':
    # localUniverse = Universe.Universe().getLocalUniverse()[:200]
    localUniverse = [
        'AAPL',
        'MSFT',
        'GOOGL',
        'INTC',
        'AMZN',
        'FB',
        'BABA',
        'JNJ',
        'JPM'
    ]
    stockPrices = [Prices.Prices(x) for x in localUniverse]
    stockPrices = [x for x in stockPrices if x.hasEnoughData()]

    returns = [x.getDataForCovarianceCalculation() for x in stockPrices]
    p = np.asmatrix([x.getAnnualMeanReturnAndVolatility()[0] for x in stockPrices])
    C = 100 * np.asmatrix(np.cov(returns)) * np.sqrt(252)

    def random_portfolio():
        n = len(returns)
        weights = np.random.rand(n)
        weights = weights / sum(weights)
        w = np.asmatrix(weights)
        mu = w * p.T
        sigma = np.sqrt(w * C * w.T)
        return float(mu[0][0]), float(sigma[0,0])

    n_portfolios = 2500
    means, stds = np.column_stack([
        random_portfolio() 
        for _ in range(n_portfolios)
    ])

    portfolios = []
    for x in range(len(means)):
        portfolios.append((means[x], stds[x]))
    portfoliosByReturn = sorted(portfolios, key = lambda tup: tup[0], reverse = True)
    portfoliosByVolatility = sorted(portfolios, key = lambda tup: tup[1])

    plt.plot(stds, means, 'o', markersize=3)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')
    plt.scatter(x=portfoliosByReturn[0][1], y=portfoliosByReturn[0][0], c='red', marker='D', s=50)
    plt.scatter(x=portfoliosByVolatility[0][1], y=portfoliosByVolatility[0][0], c='green', marker='D', s=50)
    plt.show()

    pairs = [x.getAnnualMeanReturnAndVolatility() for x in stockPrices]
    x_val = [x[1] for x in pairs]
    y_val = [x[0] for x in pairs]
    plt.scatter(x_val,y_val)
    plt.title('Return and Volatility')
    plt.xlabel('Expected Volatility (SD)')
    plt.ylabel('Expected Return')
    plt.show()