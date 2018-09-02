import matplotlib.pyplot as plt
from Shared import Prices, Universe
import numpy as np
import cvxopt as opt
from cvxopt import solvers
import statistics
import pandas as pd

# https://blog.quantopian.com/markowitz-portfolio-optimization-2/
# http://dacatay.com/data-science/portfolio-optimization-python/

class MPT:
    __USE_PREDEFINED_UNIVERSE_LIST = True
    __stockPrices = None
    __expectedReturns = None
    __covariance = None
    RISK_FREE_RATE = 6
    def __init__(self):
        if self.__USE_PREDEFINED_UNIVERSE_LIST:
            # https://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NYSE&sortname=marketcap&sorttype=1
            self.__localUniverse = [
                # 'BABA',     # Alibaba Group Holding Limited (Not enough data)
                'JPM',      # JP Morgan  
                'JNJ',      # Johnson & Johnson    
                'XOM',      # Exxon Mobile Corporation
                'V',        # Visa Inc.
                'BAC',      # Bank of America Corporation
                'WMT',      # Walmart Inc.
                'WFC',      # Wells Fargo & Company
                'UNH',      # UnitedHealth Group Incorporated
                'PFE',      # Pfizer, Inc.
                'MA'        # Mastercard Incorporated
            ]  
        else:
            self.__localUniverse = sorted(Universe.Universe().getLocalUniverse())[:10]
            
    def annualize(self, value, periods = 12):
        return 100 * np.sqrt(periods) * value   
    
    def getStocks(self):
        if self.__stockPrices is None:
            self.__stockPrices = [Prices.Prices(x) for x in self.__localUniverse]
            self.__stockPrices = [x for x in self.__stockPrices if x.hasEnoughData()]
        return self.__stockPrices
    
    def getCovariance(self):
        if self.__covariance is None:
            self.__covariance = np.asmatrix(np.cov([x.getDataForCovarianceCalculation() for x in self.getStocks()]))
        return self.__covariance
    
    def getExpectedReturns(self):
        if self.__expectedReturns is None:
            self.__expectedReturns = np.asmatrix([100 * x.getAnnualMeanReturnAndVolatility()[0] for x in self.getStocks()]) # E(r)
        #return self.__annualize(np.array([x.mean() for x in periodicReturns])) # E(r)
        return self.__expectedReturns
    
    def getRandomPortfolio(self):
        weights = np.random.rand(len(self.getStocks()))
        weights = np.asmatrix(weights / sum(weights))
        expectedReturn = float((weights * self.getExpectedReturns().T)[0][0])
        expectedVolatility = float(np.sqrt(weights * self.getCovariance() * weights.T)[0][0])
        return (expectedReturn, mpt.annualize(expectedVolatility), weights)
    
    def getMaxReturn(self):
        return min(self.getExpectedReturns().max() * 0.99, 50)
    
    def getMinReturn(self):
        return max(self.getExpectedReturns().mean(), self.RISK_FREE_RATE)
    
    def getEvenlySpacedMinimumReturns(self, numberOfEfficientPortfolios = 15):
        currentIterate = self.getMinReturn()
        maxReturn = self.getMaxReturn()
        minReturn = self.getMinReturn()
        returnRange = maxReturn - minReturn
        stepSize = returnRange / numberOfEfficientPortfolios
        
        minimumReturns = []
        for i in range(numberOfEfficientPortfolios):
            minimumReturns.append(currentIterate)
            currentIterate = currentIterate + stepSize
        
        return minimumReturns
    
    def getFrontierPortfolios(self, minimumReturns):
        portfolioSize = len(self.getStocks())
        P = opt.matrix(self.getCovariance())
        q = opt.matrix(np.zeros((portfolioSize, 1)))
        G = opt.matrix(np.concatenate((-1 * np.array(self.getExpectedReturns()), -1 * np.identity(portfolioSize)), 0))
        A = opt.matrix(1.0, (1, portfolioSize))
        b = opt.matrix(1.0)

        opt.solvers.options['show_progress'] = False
        optimalWeightsAlongTheFrontier = [solvers.qp(P, q, G, opt.matrix(np.concatenate((-np.ones((1, 1)) * mu, np.zeros((portfolioSize, 1))), 0)), A, b)['x'] for mu in minimumReturns]      
        optimalVolatilityAlongTheFrontier = [np.sqrt(np.matrix(w).T * self.getCovariance().T.dot(np.matrix(w)))[0,0] for w in optimalWeightsAlongTheFrontier]
        optimalVolatilityAlongTheFrontier = [self.annualize(x) for x in optimalVolatilityAlongTheFrontier]
        return (minimumReturns, optimalVolatilityAlongTheFrontier, optimalWeightsAlongTheFrontier)
  
    def getCapitalAllocationLine(self, expectedReturns, expectedVolatilities):
        bestIndex = 0
        maxSlope = 0
        for i in range(len(expectedReturns)):
            newMaxSlope = (expectedReturns[i] - self.RISK_FREE_RATE) / expectedVolatilities[i]
            if newMaxSlope > maxSlope:
                maxSlope = newMaxSlope
                bestIndex = i  
        return (maxSlope, bestIndex)
    
    def perturbReturns(self):
        expectedReturns = 0.01 * np.multiply(self.getExpectedReturns()[0], np.random.randint(low=95, high=105, size = len(self.getStocks())))
        self.__expectedReturns = expectedReturns
        
    def resetReturns(self):
        self.__expectedReturns = None
        
if __name__ == '__main__':
    NUMBER_OF_RANDOM_PORTFOLIOS = 2500
    
    mpt = MPT()
    randomPortfolios = [mpt.getRandomPortfolio() for x in range(NUMBER_OF_RANDOM_PORTFOLIOS)]
    randomPortfoliosExpectedReturn = [x[0] for x in randomPortfolios]
    randomPortfoliosExpectedVolatilities = [x[1] for x in randomPortfolios]
    randomPortfoliosWeights = [x[2] for x in randomPortfolios]
    
    evenlySpacedMinimumReturns = mpt.getEvenlySpacedMinimumReturns(15)
    medianMinimumReturn = np.median(mpt.getExpectedReturns(), axis = 1)[0][0] # Target
    
    optimalPortfolios = mpt.getFrontierPortfolios(medianMinimumReturn)
    optimalPortfoliosExpectedReturn = optimalPortfolios[0]
    optimalPortfoliosExpectedVolatilities = optimalPortfolios[1]
    optimalPortfoliosWeights = opt.matrix(optimalPortfolios[2])
    
    allocations = [optimalPortfoliosWeights]

    for i in range(80):  
        mpt.perturbReturns()
        optimalPortfolios = mpt.getFrontierPortfolios(medianMinimumReturn)
        allocations.append(opt.matrix(optimalPortfolios[2]))
        mpt.resetReturns()
    
    toplot = []
    for weights in allocations:
        for index, weight in enumerate(weights):
            toplot.append((index + 1,weight))
    plt.plot([x[0] for x in toplot], [x[1] for x in toplot], 'o', markersize = 3) 
    plt.show()
    toplot = pd.DataFrame(toplot, columns = ['Stock', 'Weight'])
    
    for stock in toplot.Stock.unique():
        stockPoints = toplot.loc[toplot['Stock'] == stock]['Weight'].tolist()
        plt.hist(stockPoints)
        plt.show()
    
#    print(toplot.groupby(['Stock']).mean())
#    print(toplot.groupby(['Stock']).agg(np.std))
    
#    cal = mpt.getCapitalAllocationLine(optimalPortfoliosExpectedReturn, optimalPortfoliosExpectedVolatilities)
#    plt.plot(randomPortfoliosExpectedVolatilities, randomPortfoliosExpectedReturn, 'o', markersize = 3)
#    plt.plot(optimalPortfoliosExpectedVolatilities, optimalPortfoliosExpectedReturn, 'y-o', color='orange', markersize=5, label='Efficient Frontier')
##    plt.plot([0, max(optimalPortfoliosExpectedVolatilities)],[mpt.RISK_FREE_RATE, cal[0] * max(optimalPortfoliosExpectedVolatilities) + mpt.RISK_FREE_RATE],'k-', label='Capital Allocation Line')
##    plt.plot([optimalPortfoliosExpectedVolatilities[cal[1]]], [optimalPortfoliosExpectedReturn[cal[1]]], color='green', markersize=10, label='Optimal Portfolio')
##    plt.plot([0], [mpt.RISK_FREE_RATE], color='green', markersize=10)
#    plt.xlabel('Expected Portfolio Volatility (Standard deviation of returns, %)')
#    plt.ylabel('Expected Portfolio Annual Return (%)')
#    plt.title('Mean and standard deviation of returns portfolios')
##    plt.axhline(y=mpt.getExpectedReturns().mean(), color='r', linestyle='-')
##    plt.xlim(0, max(optimalPortfoliosExpectedVolatilities) + 5)
##    plt.ylim(mpt.RISK_FREE_RATE - 1, mpt.getMaxReturn())
#    plt.show()