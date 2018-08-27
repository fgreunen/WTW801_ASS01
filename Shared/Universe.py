import pandas as pd
import glob, os

class Universe:
    def __init__(self, datadir = './Data/'):
        df = pd.read_csv(os.path.join(datadir, 'WIKI-datasets-codes.csv'), delimiter=',')
        self._universe = [tuple(x) for x in df.values]
        self.DATA_DIR = datadir
    
    def getUniverse(self):
        return [str(x[0])[5:] for x in self._universe]

    def getLocalUniverse(self):
        files = [x.split('-')[0] for x in os.listdir(self.DATA_DIR) if x.endswith(".pkl")]
        return list(set(files))