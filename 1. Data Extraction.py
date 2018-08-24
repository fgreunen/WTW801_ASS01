from Shared import Prices
from Shared import Universe

if __name__ == '__main__':
    universe = Universe.Universe().getUniverse()
    print('Universe loaded: {0} stocks'.format(len(universe)))
    for index, stock in enumerate(universe):
        print('Retrieving {0}...'.format(stock) )
        pricesRef = Prices.Prices(stock)
        print('{0} of {1}'.format(index + 1, len(universe)))