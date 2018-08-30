import matplotlib.pyplot as plt
import matplotlib as mpl
from Shared import Prices, Universe
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxopt as opt
from cvxopt import solvers


# https://blog.quantopian.com/markowitz-portfolio-optimization-2/
# http://dacatay.com/data-science/portfolio-optimization-python/

USE_DAILY_RETURNS_FOR_EXPECTED_RETURN = False
RISK_FREE_RATE = 6

if __name__ == '__main__':
    localUniverse = sorted(Universe.Universe().getLocalUniverse())[:10]
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
        expectedReturns = [100 * x.getAnnualMeanReturnAndVolatility()[0] for x in stockPrices] # E(r)

    portfolioSize = len(expectedReturns)
    expectedReturns = np.asmatrix(expectedReturns)
    covariance = np.asmatrix(np.cov(periodicReturns)) # Covariance matrix

    def random_portfolio():
        weights = np.random.rand(portfolioSize)
        weights = weights / sum(weights) # Explicitly scale the asset weights to have a sum of one
        weights = np.asmatrix(weights)
        mu = weights * expectedReturns.T # Expected return for the entire portfolio with the defined asset weights
        mu = float(mu[0][0])
        sigma = np.sqrt(weights * covariance * weights.T)
        sigma = annualize(float(sigma[0,0]))
        return mu, sigma

    n_portfolios = 200
    means, stds = np.column_stack([
        random_portfolio() 
        for _ in range(n_portfolios)
    ])

    portfolios = []
    for x in range(len(means)):
        portfolios.append((means[x], stds[x]))

    portfolios = [x for x in portfolios if x[0] > 3]
    
    maxReturn = min(expectedReturns.max() * 0.99, 50) # To ensure a solution.
    minReturn = max(expectedReturns.mean(), RISK_FREE_RATE)
    numberOfEfficientPortfolios = 15
    returnRange = maxReturn - minReturn
    stepSize = returnRange / numberOfEfficientPortfolios

    def calculate_frontier():

        currentIterate = minReturn
        optimal_mus = [] # minimum expected return
        for i in range(numberOfEfficientPortfolios):
            optimal_mus.append(currentIterate)
            currentIterate = currentIterate + stepSize
        
        # constraint matrices for quadratic programming
        P = opt.matrix(covariance)
        q = opt.matrix(np.zeros((portfolioSize, 1)))
        G = opt.matrix(np.concatenate((-1 * np.array(expectedReturns), -1 * np.identity(portfolioSize)), 0))
        A = opt.matrix(1.0, (1, portfolioSize))
        b = opt.matrix(1.0)
        
        # hide optimization
        opt.solvers.options['show_progress'] = False
        
        optimal_weights = [solvers.qp(P, q, G, opt.matrix(np.concatenate((-np.ones((1, 1)) * mu, np.zeros((portfolioSize, 1))), 0)), A, b)['x'] for mu in optimal_mus]
        
        # find optimal sigma
        # \sigma = w^T * Cov * w
        optimal_sigmas = [np.sqrt(np.matrix(w).T * covariance.T.dot(np.matrix(w)))[0,0] for w in optimal_weights]
        optimal_sigmas = [annualize(x) for x in optimal_sigmas ]
        return optimal_weights, optimal_mus, optimal_sigmas
    optimal_weights, optimal_mus, optimal_sigmas = calculate_frontier()

    bestIndex = 0
    maxSlope = 0
    for i in range(len(optimal_mus)):
        newMaxSlope = (optimal_mus[i] - RISK_FREE_RATE) / optimal_sigmas[i]
        if newMaxSlope > maxSlope:
            maxSlope = newMaxSlope
            bestIndex = i

    plt.plot([x[1] for x in portfolios], [x[0] for x in portfolios], 'o', markersize=3)
    plt.plot(optimal_sigmas, optimal_mus, 'y-o', color='orange', markersize=5, label='Efficient Frontier')
    plt.plot([0, max(optimal_sigmas)],[RISK_FREE_RATE, maxSlope * max(optimal_sigmas) + RISK_FREE_RATE],'k-', label='Capital Allocation Line')
    plt.plot([optimal_sigmas[bestIndex]], [optimal_mus[bestIndex]], color='green', markersize=10, label='Optimal Portfolio')
    plt.plot([0], [RISK_FREE_RATE], color='green', markersize=10)
    plt.xlabel('Expected Portfolio Volatility (Standard deviation of returns, %)')
    plt.ylabel('Expected Portfolio Annual Return (%)')
    plt.title('Mean and standard deviation of returns portfolios')
    plt.axhline(y=expectedReturns.mean(), color='r', linestyle='-')
    plt.xlim(0, 25)
    plt.ylim(RISK_FREE_RATE - 1, maxReturn)
    plt.show()