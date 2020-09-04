import os
import sklearn.preprocessing

class DataHandler:
    data_dir = '../data'
    def __init__(self, fname):
        self.csv = os.path.join(self.data_dir, fname)

    def one_hot_encode(self, df) -> None:
        _one_hot_encode(df, self.one_hot_vars)

class AbaloneHandler(DataHandler):
    def __init__(self):
        self.one_hot_vars = ['sex']
        super().__init__('abalone.csv')

def _one_hot_encode(df, one_hot_vars):
    encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    encoder.fit(df[one_hot_vars])
    colnames = list(encoder.categories_[0])
    one_hot_cols = encoder.transform(df[one_hot_vars])
    df[colnames] = one_hot_cols
    df.drop(columns=one_hot_vars, inplace=True)
    
    
