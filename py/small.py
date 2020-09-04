import os
from sklearn import preprocessing, model_selection
import pandas as pd
from enum import Enum
import random
from typing import Dict, Any

class Splits(Enum):
    trn = 'train'
    val = 'validation'
    tst = 'test'

class DataHandler:
    data_dir = '../data'
    data_props = {Splits.trn: 0.8, Splits.val: 0.5, Splits.tst: 0.5}
    def __init__(self, fname):
        self.csv = os.path.join(self.data_dir, fname)

    def read(self):
        df = pd.read_csv(self.csv)
        self.original_predictors = [col for col in df.columns
                                    if col != self.response_var]
        return df

    def one_hot_encode(self, df) -> pd.DataFrame:
        """
        One hot encodes the one_hot_vars columns in df.
        Remembers the fit encoder.
        """
        df = df.copy()
        try:
            self.encoder
        except AttributeError:
            self.encoder = preprocessing.OneHotEncoder(sparse=False)
            self.encoder.fit(df[self.one_hot_vars])
        colnames = list(self.encoder.categories_[0])
        one_hot_cols = self.encoder.transform(df[self.one_hot_vars])
        df[colnames] = one_hot_cols
        df = df.drop(columns=self.one_hot_vars)
        self.encoded_predictors = [col for col in df.columns
                                   if col != self.response_var]
        return df

    def get_predictor_colnames(self):
        try:
            return self.encoded_predictors
        except AttributeError:
            return self.original_predictors
    
    def standardize(self, df):
        """
        Standardizes all columns in df using StandardScaler.
        Remembers the fit standardizer.
        """
        df = df.copy()
        scalecols = df.columns
        try:
            self.standardizer
        except AttributeError:
            self.standardizer = preprocessing.StandardScaler()
            self.standardizer.fit(df[scalecols])        
        df[scalecols] = self.standardizer.transform(df[scalecols])
        return df

    def split_predictors_response(self, df):
        predictors = self.get_predictor_colnames()
        X = df[predictors]
        y = df[self.response_var]
        return X, y

    def split_and_apply_transforms(self, df) -> Dict[str, Dict[str, Any]]:
        datasets = {}
        trn, val, tst = self.split(df)
        for name, subdf in zip(Splits, (trn, val, tst)):
            X, y = self.apply_transforms(subdf)
            subdict = {'X': X, 'y': y, 'data': subdf}            
            datasets[name] = subdict
        return datasets

    def apply_transforms(self, df):
        df = self.one_hot_encode(df)
        df = self.standardize(df)
        X, y = self.split_predictors_response(df)
        return X.values, y.values
    
    def split(self, df):
        random.seed(4)
        trn, valtst = model_selection.train_test_split(
            df, test_size = 1 - self.data_props[Splits.trn])
        val, tst = model_selection.train_test_split(
            valtst, test_size = self.data_props[Splits.tst])
        return trn, val, tst

class AbaloneHandler(DataHandler):
    def __init__(self):
        self.one_hot_vars = ['sex']
        self.response_var = 'rings'
        super().__init__('abalone.csv')

