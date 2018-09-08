import matplotlib.pyplot as plt
from Shared import Prices
from datetime import timedelta
from prettytable import PrettyTable
import pandas as pd
from scipy import stats
import numpy as np
import cvxopt as opt
from cvxopt import solvers

if __name__ == '__main__':
    totalReturnsOverEntirePeriod = []
    yearlyRateOfReturns = []
    
    tickername = 'JPM'
    prices = Prices.Prices(tickername)
    yearlyRateOfReturns.append((tickername, prices.getYearlyRateOfReturns()))
    prices = prices.getPrices()
    first = prices.index[0]
    last = prices.index[len(prices.index) - 1]
    totalReturnsOverEntirePeriod.append((tickername, prices['adj_close'][len(prices.index) - 1] / prices['adj_close'][0]))
    xPlotMin = min(prices.index) - timedelta(days=350)
    xPlotMax = max(prices.index) + timedelta(days=240)
    prices.plot(y='adj_close', legend=None, color='#333333')
    plt.title(f'Close Price (Adjusted) for \nJP Morgan ({tickername})')
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylim(min(prices['adj_close']) - 10, max(prices['adj_close']) + 14)
    plt.xlim(xPlotMin, xPlotMax)
    plt.annotate(first.date(), [first - timedelta(days=280), prices['adj_close'][0] + 8])
    plt.annotate(last.date(), [last - timedelta(days=400), prices['adj_close'][len(prices.index) - 1] + 8])
    plt.plot(first.date(), prices['adj_close'][0], 'o', markersize = 10, color='b')
    plt.plot(last.date(), prices['adj_close'][len(prices.index) - 1], 'o', markersize = 10, color='g')
    plt.hlines(y=prices['adj_close'][0], xmin = xPlotMin, xmax = first.date(), color='b', linestyle='--')
    plt.hlines(y=prices['adj_close'][len(prices.index) - 1], xmin = xPlotMin, xmax = last.date(), color='g', linestyle='--')
    plt.show()

    tickername = 'JNJ'
    prices = Prices.Prices(tickername)
    yearlyRateOfReturns.append((tickername, prices.getYearlyRateOfReturns()))
    prices = prices.getPrices()
    first = prices.index[0]
    last = prices.index[len(prices.index) - 1]
    totalReturnsOverEntirePeriod.append((tickername, prices['adj_close'][len(prices.index) - 1] / prices['adj_close'][0]))
    xPlotMin = min(prices.index) - timedelta(days=350)
    xPlotMax = max(prices.index) + timedelta(days=240)
    prices.plot(y='adj_close', legend=None, color='#333333')
    plt.title(f'Close Price (Adjusted) for \nJohnson & Johnson ({tickername})')
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylim(min(prices['adj_close']) - 10, max(prices['adj_close']) + 14)
    plt.xlim(xPlotMin, xPlotMax)
    plt.annotate(first.date(), [first - timedelta(days=280), prices['adj_close'][0] + 8])
    plt.annotate(last.date(), [last - timedelta(days=400), prices['adj_close'][len(prices.index) - 1] + 8])
    plt.plot(first.date(), prices['adj_close'][0], 'o', markersize = 10, color='b')
    plt.plot(last.date(), prices['adj_close'][len(prices.index) - 1], 'o', markersize = 10, color='g')
    plt.hlines(y=prices['adj_close'][0], xmin = xPlotMin, xmax = first.date(), color='b', linestyle='--')
    plt.hlines(y=prices['adj_close'][len(prices.index) - 1], xmin = xPlotMin, xmax = last.date(), color='g', linestyle='--')
    plt.show()
    
    tickername = 'XOM'
    prices = Prices.Prices(tickername)
    yearlyRateOfReturns.append((tickername, prices.getYearlyRateOfReturns()))
    prices = prices.getPrices()
    first = prices.index[0]
    last = prices.index[len(prices.index) - 1]
    totalReturnsOverEntirePeriod.append((tickername, prices['adj_close'][len(prices.index) - 1] / prices['adj_close'][0]))
    xPlotMin = min(prices.index) - timedelta(days=350)
    xPlotMax = max(prices.index) + timedelta(days=240)
    prices.plot(y='adj_close', legend=None, color='#333333')
    plt.title(f'Close Price (Adjusted) for \nExxon Mobile Corporation ({tickername})')
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylim(min(prices['adj_close']) - 7, max(prices['adj_close']) + 10)
    plt.xlim(xPlotMin, xPlotMax)
    plt.annotate(first.date(), [first - timedelta(days=312), prices['adj_close'][0] + 6])
    plt.annotate(last.date(), [last - timedelta(days=400), prices['adj_close'][len(prices.index) - 1] + 7])
    plt.plot(first.date(), prices['adj_close'][0], 'o', markersize = 10, color='b')
    plt.plot(last.date(), prices['adj_close'][len(prices.index) - 1], 'o', markersize = 10, color='g')
    plt.hlines(y=prices['adj_close'][0], xmin = xPlotMin, xmax = first.date(), color='b', linestyle='--')
    plt.hlines(y=prices['adj_close'][len(prices.index) - 1], xmin = xPlotMin, xmax = last.date(), color='g', linestyle='--')
    plt.show()
    
    tickername = 'V'
    prices = Prices.Prices(tickername)
    yearlyRateOfReturns.append((tickername, prices.getYearlyRateOfReturns()))
    prices = prices.getPrices()
    first = prices.index[0]
    last = prices.index[len(prices.index) - 1]
    totalReturnsOverEntirePeriod.append((tickername, prices['adj_close'][len(prices.index) - 1] / prices['adj_close'][0]))
    xPlotMin = min(prices.index) - timedelta(days=350)
    xPlotMax = max(prices.index) + timedelta(days=240)
    prices.plot(y='adj_close', legend=None, color='#333333')
    plt.title(f'Close Price (Adjusted) for \nVisa Inc. ({tickername})')
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylim(min(prices['adj_close']) - 10, max(prices['adj_close']) + 18)
    plt.xlim(xPlotMin, xPlotMax)
    plt.annotate(first.date(), [first - timedelta(days=280), prices['adj_close'][0] + 8])
    plt.annotate(last.date(), [last - timedelta(days=400), prices['adj_close'][len(prices.index) - 1] + 8])
    plt.plot(first.date(), prices['adj_close'][0], 'o', markersize = 10, color='b')
    plt.plot(last.date(), prices['adj_close'][len(prices.index) - 1], 'o', markersize = 10, color='g')
    plt.hlines(y=prices['adj_close'][0], xmin = xPlotMin, xmax = first.date(), color='b', linestyle='--')
    plt.hlines(y=prices['adj_close'][len(prices.index) - 1], xmin = xPlotMin, xmax = last.date(), color='g', linestyle='--')
    plt.show()
    
    tickername = 'BAC'
    prices = Prices.Prices(tickername)
    yearlyRateOfReturns.append((tickername, prices.getYearlyRateOfReturns()))
    prices = prices.getPrices()
    first = prices.index[0]
    last = prices.index[len(prices.index) - 1]
    totalReturnsOverEntirePeriod.append((tickername, prices['adj_close'][len(prices.index) - 1] / prices['adj_close'][0]))
    xPlotMin = min(prices.index) - timedelta(days=350)
    xPlotMax = max(prices.index) + timedelta(days=240)
    prices.plot(y='adj_close', legend=None, color='#333333')
    plt.title(f'Close Price (Adjusted) for \nBank of America Corporation ({tickername})')
    plt.ylabel('Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=90)
    plt.ylim(min(prices['adj_close']) - 1, max(prices['adj_close']) + 7)
    plt.xlim(xPlotMin, xPlotMax)
    plt.annotate(first.date(), [first - timedelta(days=280), prices['adj_close'][0] + 5])
    plt.annotate(last.date(), [last - timedelta(days=400), prices['adj_close'][len(prices.index) - 1] + 2])
    plt.plot(first.date(), prices['adj_close'][0], 'o', markersize = 10, color='b')
    plt.plot(last.date(), prices['adj_close'][len(prices.index) - 1], 'o', markersize = 10, color='g')
    plt.hlines(y=prices['adj_close'][0], xmin = xPlotMin, xmax = first.date(), color='b', linestyle='--')
    plt.hlines(y=prices['adj_close'][len(prices.index) - 1], xmin = xPlotMin, xmax = last.date(), color='g', linestyle='--')
    plt.show()
    
    names = [t[0] for t in totalReturnsOverEntirePeriod]
    values = [t[1] for t in totalReturnsOverEntirePeriod]
    plt.bar(names, values, color='#333333')
    plt.title(f'Total Return of assets \nover the entire period (2010 - 2017)')
    plt.ylabel('Total Return')
    plt.xlabel('Asset')
    plt.show()
    
    headers = ['Year']
    headers.extend([x[0] for x in yearlyRateOfReturns])
    iterate = [x[1] for x in yearlyRateOfReturns]
    stocks = len(iterate)
    rows = []
    strRows = []
    for j in range(len(iterate[0])):
        row = [iterate[0][j][1]]
        strRow = [iterate[0][j][1]]
        for i in range(stocks):
            row.append(iterate[i][j][0])
            strRow.append(f'{100 * iterate[i][j][0]:05.2f}%')
        rows.append(row)
        strRows.append(strRow)
        
    df = pd.DataFrame(data = rows, columns = headers)
    df.set_index('Year', inplace=True)
    
    t = PrettyTable(headers)
    for i in range(len(strRows)):
        t.add_row(strRows[i])
    
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
    
    covariance = np.asmatrix(df.cov())
    returns = np.asmatrix([(stats.gmean(df[x] + 1) - 1) for x in df])
    
    minimumReturns = [0.06]
    for i in range(20):
        minimumReturns.append(round(minimumReturns[len(minimumReturns) - 1] + 0.005, 3))
        
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
    columns.extend([x[0] for x in yearlyRateOfReturns])
    optimalPortfoliosDf = pd.DataFrame(data = optimalPortfoliosExplodedWeights, 
                                       columns = columns)
    optimalPortfoliosDf = optimalPortfoliosDf.set_index('Variance')
    
    print(optimalPortfoliosDf)
    
    optimalPortfoliosDf.plot(y='ExpectedReturn', legend=None, color='#333333')   
    plt.plot(optimalPortfoliosDf.index, optimalPortfoliosDf['ExpectedReturn'], 'o', markersize = 5, color='g')
    plt.title(f'Volatility versus Expected Return \nof portfolios along the efficient frontier')
    plt.ylabel('Expected Return (%)')
    plt.xlabel('Standard Deviation (%)')
    plt.ylim(min(optimalPortfoliosDf['ExpectedReturn']), max(optimalPortfoliosDf['ExpectedReturn']) + 1)
    plt.show()

    headers = optimalPortfoliosDf.columns.values.tolist()
    headers.insert(0, 'Variance')
    t = PrettyTable(headers)
    for index, row in optimalPortfoliosDf.iterrows():
        toPrint = [f'{index:05.2f}%']
        for column in optimalPortfoliosDf: 
            toPrint.append(f'{optimalPortfoliosDf[column][index]:05.2f}%')
        t.add_row(toPrint)
    print(t)
    
    
    
    
    
    
    
    