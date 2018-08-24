import pandas as pd
import os

class Universe:
    def __init__(self, datadir = './Data/'):
        df = pd.read_csv(os.path.join(datadir, 'WIKI-datasets-codes.csv'), delimiter=',')
        self._universe = [tuple(x) for x in df.values]
    
    def getUniverse(self):
        return [str(x[0])[5:] for x in self._universe]