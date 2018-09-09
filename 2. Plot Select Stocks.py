import matplotlib.pyplot as plt
from Shared import Prices, Universe
from datetime import timedelta
from prettytable import PrettyTable
import pandas as pd
from scipy import stats
import numpy as np
import cvxopt as opt
from cvxopt import solvers

fixedstockuniverse = stockUniverse = ['JPM', 'JNJ', 'XOM', 'V', 'BAC', 'WMT', 'WFC', 'UNH', 'PFE', 'MA']
#stockUniverse = sorted(Universe.Universe().getLocalUniverse())[:1000]
prices = [Prices.Prices(tickername) for tickername in stockUniverse]
prices = [x for x in prices if x.isFromStart()]
print(len(prices))

def plotAdjustedClose(tickername, name, p1 = 350, p2 = 240, p3 = 10, p4 = 14, 
                      p5 = 280, p6 = 8, p7 = 400, p8 = 8):
    prices = Prices.Prices(tickername).getPrices()
    first = prices.index[0]
    last = prices.index[len(prices.index) - 1]
    xPlotMin = min(prices.index) - timedelta(days=p1)
    xPlotMax = max(prices.index) + timedelta(days=p2)
    prices.plot(y='adj_close', legend=None, color='#333333')
#    plt.title(f'Close Price (Adjusted) for \n{name} ({tickername})')
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylim(min(prices['adj_close']) - p3, max(prices['adj_close']) + p4)
    plt.xlim(xPlotMin, xPlotMax)
    plt.annotate(first.date(), [first - timedelta(days=p5), prices['adj_close'][0] + p6])
    plt.annotate(last.date(), [last - timedelta(days=p7), prices['adj_close'][len(prices.index) - 1] + p8])
    plt.plot(first.date(), prices['adj_close'][0], 'o', markersize = 10, color='b')
    plt.plot(last.date(), prices['adj_close'][len(prices.index) - 1], 'o', markersize = 10, color='g')
    plt.hlines(y=prices['adj_close'][0], xmin = xPlotMin, xmax = first.date(), color='b', linestyle='--')
    plt.hlines(y=prices['adj_close'][len(prices.index) - 1], xmin = xPlotMin, xmax = last.date(), color='g', linestyle='--')
    plt.show()
    
def plotTotalReturns():
    totalReturnsOverEntirePeriod = [(x.tickername, x.getPrices()) for x in prices]
    totalReturnsOverEntirePeriod = [(x[0], x[1]['adj_close'][len(x[1].index) - 1] / x[1]['adj_close'][0]) for x in totalReturnsOverEntirePeriod]

    names = [t[0] for t in totalReturnsOverEntirePeriod]
    values = [t[1] for t in totalReturnsOverEntirePeriod]
    plt.bar(names, values, color='#333333')
#    plt.title(f'Total Return of assets \nover the entire period (2010 - 2017)')
    plt.ylabel('Total Return')
    plt.xlabel('Asset')
    plt.show()
    
def getYearlyRateOfReturns():
    yearlyRateOfReturns = [(x.tickername, x.getYearlyRateOfReturns()) for x in prices]
    headers = ['Year']
    headers.extend([x[0] for x in yearlyRateOfReturns])
    iterate = [x[1] for x in yearlyRateOfReturns]
    stocks = len(iterate)
    rows = []
    for j in range(len(iterate[0])):
        row = [iterate[0][j][1]]
        for i in range(stocks):
            row.append(iterate[i][j][0])
        rows.append(row)
    df = pd.DataFrame(data = rows, columns = headers)
    df.set_index('Year', inplace=True)
    return df

def printYearlyReturnsTable(df):
    headers = ['Year']
    headers.extend(df.columns.values.tolist())
    
    t = PrettyTable(headers)
    for index, row in df.iterrows():
        toPrint = [f'{index}']
        for column in df: 
            toPrint.append(f'{100 * df[column][index]:05.2f}%')
        t.add_row(toPrint)
    
    arithmeticMeans = [f'{100 * x:05.2f}%' for x in df.mean(axis=0).tolist()]
    arithmeticMeans.insert(0, 'Arithmetic Mean')
    t.add_row(arithmeticMeans)

    geometricMeans = [f'{(stats.gmean(df[x] + 1) - 1) * 100:05.2f}%' for x in df]
    geometricMeans.insert(0, 'Geometric Mean')
    t.add_row(geometricMeans)   
    
    volatility = [f'{100 * x:05.2f}%' for x in df.std()]
    volatility.insert(0, 'Volatility (STD)')
    t.add_row(volatility)  
    
    print(t)
    
    
def getFrontierPortfolios(df, perturbReturns = False, useMedian = False):
    covariance = np.asmatrix(df.cov())
    returns = np.asmatrix([(stats.gmean(df[x] + 1) - 1) for x in df])
    if useMedian:
        minimumReturns = [np.median(np.array(returns))]
    else:
        minimumReturns = [0.07]
        for i in range(20):
            minimumReturns.append(round(minimumReturns[len(minimumReturns) - 1] + 0.01, 3))
    
    if perturbReturns:
        randomMatrix = np.random.uniform(95,106,returns.size)
        returns = 0.01 * np.multiply(returns, randomMatrix)
    
    portfolioSize = returns.size
    P = opt.matrix(covariance)
    q = opt.matrix(np.zeros((portfolioSize, 1)))
    G = opt.matrix(np.concatenate((-1 * np.array(returns), -1 * np.identity(portfolioSize)), 0))
    A = opt.matrix(1.0, (1, portfolioSize))
    b = opt.matrix(1.0)

    opt.solvers.options['show_progress'] = False
    weights = [solvers.qp(P, q, G, opt.matrix(np.concatenate((-np.ones((1, 1)) * mu, np.zeros((portfolioSize, 1))), 0)), A, b)['x'] for mu in minimumReturns]      
    volatility = [np.sqrt(np.matrix(w).T * covariance.T.dot(np.matrix(w)))[0,0] for w in weights]
    
    optimalPortfolios = []
    for i in range(len(minimumReturns)):
        optimalPortfolios.append((minimumReturns[i] * 100, volatility[i] * 100, weights[i]))

    optimalPortfoliosExplodedWeights = []
    for i in range(len(optimalPortfolios)):
        newRow = [optimalPortfolios[i][0], optimalPortfolios[i][1]]
        newRow.extend(optimalPortfolios[i][2] * 100)
        optimalPortfoliosExplodedWeights.append(newRow)
        
    columns = ['ExpectedReturn', 'Variance']
    columns.extend([x for x in df.loc[:, df.columns != 'ExpectedReturn'].columns.values.tolist()])
    optimalPortfoliosDf = pd.DataFrame(data = optimalPortfoliosExplodedWeights, columns = columns) 
    return optimalPortfoliosDf

def plotFrontierPortfolios(frontierPortfolios):
    frontierPortfolios.plot(y='ExpectedReturn',x = 'Variance', legend=None, color='#333333')   
    plt.plot(frontierPortfolios['Variance'], frontierPortfolios['ExpectedReturn'], 'o', markersize = 5, color='g')
#    plt.title(f'Volatility versus Expected Return \nof portfolios along the efficient frontier')
    plt.ylabel('Expected Return (%)')
    plt.xlabel('Standard Deviation (%)')
    plt.ylim(min(frontierPortfolios['ExpectedReturn']), max(frontierPortfolios['ExpectedReturn']) + 1)
    plt.show()
    
def printFrontierPortfolios(frontierPortfolios):
    headers = frontierPortfolios.columns.values.tolist()
    t = PrettyTable(headers)
    for i in range(len(frontierPortfolios)):
        toPrint = []
        for column in frontierPortfolios: 
            toPrint.append(f'{frontierPortfolios[column][i]:05.2f}%')
        t.add_row(toPrint)
    
    
    meanWeights = [f'{x:05.2f}%' for x in frontierPortfolios.loc[:, ~frontierPortfolios.columns.isin(['ExpectedReturn', 'Variance'])].mean(axis=0).values.tolist()]
    meanWeights.insert(0, 'Mean')
    meanWeights.insert(0, '')
    t.add_row(meanWeights)
    
    sdWeights = [f'{x:05.2f}%' for x in frontierPortfolios.loc[:, ~frontierPortfolios.columns.isin(['ExpectedReturn', 'Variance'])].std(axis=0).values.tolist()]
    sdWeights.insert(0, 'SD')
    sdWeights.insert(0, '')
    t.add_row(sdWeights)
    print(t)

def plotFrontierPortfolioWeights(frontierPortfolios):
    columns = [x for x in frontierPortfolios.loc[:, ~frontierPortfolios.columns.isin(['ExpectedReturn', 'Variance'])]]
    
    for column in columns:
        plt.plot(frontierPortfolios["ExpectedReturn"], frontierPortfolios[column], '-o')
#        plt.title(f'Weight Allocation to \'{column}\' vs Expected Return')
        print(f'Weight Allocation to \'{column}\' vs Expected Return')
        plt.ylabel('Weight Allocation (%)')
        plt.xlabel('Expected Return (%)')
        plt.show()
        
def plotFrontierPortfolioWeightsAveraged(frontierPortfolios):
    columns = [x for x in frontierPortfolios.loc[:, ~frontierPortfolios.columns.isin(['ExpectedReturn', 'Variance'])]]
    weights = []
    for column in columns:
        weights.append((column, frontierPortfolios[column].mean()))

    names = [x[0] for x in weights]
    values = [x[1] for x in weights]
    plt.bar(names, values, color='#333333')
    plt.ylabel('Average Weight Allocation (%)')
    plt.xlabel('Asset')
    plt.show()

if __name__ == '__main__':
#    plotAdjustedClose(fixedstockuniverse[0], 'JP Morgan')
#    plotAdjustedClose(fixedstockuniverse[1], 'Johnson & Johnson')
#    plotAdjustedClose(fixedstockuniverse[2], 'Exxon Mobile Corporation', p3 = 7, p4 = 10, p5 = 312, p6 = 6, p8 = 7)
#    plotAdjustedClose(fixedstockuniverse[3], 'Visa Inc.', p4 = 18)
#    plotAdjustedClose(fixedstockuniverse[4], 'Bank of America Corporation', p3 = 1, p4 = 7, p6 = 5, p8 = 2)
#    plotAdjustedClose(fixedstockuniverse[5], 'Walmart Inc.', p4 = 18)
#    plotAdjustedClose(fixedstockuniverse[6], 'Wells Fargo & Company', p4 = 18)
#    plotAdjustedClose(fixedstockuniverse[7], 'UnitedHealth Group Incorporated', p4 = 27, p6 = 15, p8 = 15)
#    plotAdjustedClose(fixedstockuniverse[8], 'Pfizer, Inc.', p3 = 1.5, p4 = 5, p6 = 2.5, p8 = 2.5)
#    plotAdjustedClose(fixedstockuniverse[9], 'Mastercard Incorporated', p4 = 22, p6 = 10, p8 = 10)

#    plotTotalReturns()
    df = getYearlyRateOfReturns()
#    printYearlyReturnsTable(df)
   
    frontierPortfolios = getFrontierPortfolios(df)
#    printFrontierPortfolios(frontierPortfolios)
#    plotFrontierPortfolios(frontierPortfolios)
#    plotFrontierPortfolioWeights(frontierPortfolios)
#    plotFrontierPortfolioWeightsAveraged(frontierPortfolios)
    
    
    frontierPortfolios = getFrontierPortfolios(df, False, True)
    plotFrontierPortfolioWeightsAveraged(frontierPortfolios)
    for i in range(500):
        frontierPortfolios = frontierPortfolios.append(getFrontierPortfolios(df, True, True))
    plotFrontierPortfolioWeightsAveraged(frontierPortfolios)
    
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    frontierPortfolios.mean()
    frontierPortfolios.std()

#    frontierPortfolios = getFrontierPortfolios(df, True)
#    frontierPortfolios = getFrontierPortfolios(df, True)