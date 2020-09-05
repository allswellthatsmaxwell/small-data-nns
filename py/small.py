import os
import pandas as pd
import numpy as np
from enum import Enum
import random
from typing import Dict, Any, List, Tuple, Callable

from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import keras
from keras.layers import Layer

class Keynum(Enum):
    """ An enum that works as a dict key. """
    def __eq__(self, e2):
        return self.name == e2.name

    def __ne__(self, obj):
        return not self.__eq__(obj)

    def __hash__(self):
        return self.name.__hash__()


class Splits(Keynum):
    trn = 'train'
    val = 'validation'
    tst = 'test'

class Regressor(Keynum):
    Linear = LinearRegression
    RandomForest = RandomForestRegressor
    XGBoost = xgb.sklearn.XGBRegressor

class Evaluator:
    def __init__(self, datasets, metric: Callable):
        self.trn = datasets[Splits.trn]
        self.val = datasets[Splits.val]
        self.tst = datasets[Splits.tst]
        self.metric = metric

class RegressionEvaluator(Evaluator):        
    def evaluate(self, model) -> float:
        """
        :param model_fn: the model to use; e.g. LinearRegression.
        Must implement fit(self, X, y).
        """
        model.fit(self.trn['X'], self.trn['y'])
        pred = model.predict(self.val['X'])
        return self.metric(self.val['y'], pred)
    
class DataHandler:
    data_dir = '../data'
    data_props = {Splits.trn: 0.6, Splits.val: 0.5, Splits.tst: 0.5}
    def __init__(self, fname):
        self.csv = os.path.join(self.data_dir, fname)

    def read(self) -> pd.DataFrame:
        """
        Reads this handler's data file from disk. 
        """
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
        colnames = np.ravel(
            [_prefix_elements(lst, col)
             for lst, col in zip(self.encoder.categories_, self.one_hot_vars)])
        one_hot_cols = self.encoder.transform(df[self.one_hot_vars])
        df[colnames] = one_hot_cols
        df = df.drop(columns=self.one_hot_vars)
        self.encoded_predictors = [col for col in df.columns
                                   if col != self.response_var]
        return df

    def get_predictor_colnames(self) -> List[str]:
        """ returns the predictors that match the columns (including order)
        of the numpy matrix X returned elsewhere. """
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
        scalecols = self.get_predictor_colnames() ## df.columns
        try:
            self.standardizer
        except AttributeError:
            self.standardizer = preprocessing.StandardScaler()
            self.standardizer.fit(df[scalecols])        
        df[scalecols] = self.standardizer.transform(df[scalecols])
        return df

    def split_predictors_response(self, df) -> Tuple[pd.DataFrame, pd.Series]:
        """ returns the predictor matrix and response series, as pandas objects. """
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

    def apply_transforms(self, df) -> Tuple[np.array, np.array]:
        """
        Applies one-hot-encoding and scaling to the variables of this data,
        and returns the predictor matrix and response array as numpy arrays 
        (2D and 1D, respectively).
        """
        df = self.one_hot_encode(df)
        df = self.standardize(df)
        X, y = self.split_predictors_response(df)
        return X.values, y.values
    
    def split(self, df, seed=4):
        """
        Splits df by row into train, validation, and test sets, using
        the proportions in the Splits enum.
        """
        trn, valtst = model_selection.train_test_split(
            df, test_size = 1 - self.data_props[Splits.trn],
            random_state=seed)
        val, tst = model_selection.train_test_split(
            valtst, test_size = self.data_props[Splits.tst],
            random_state=seed)
        return trn, val, tst

class AbaloneHandler(DataHandler):
    """
    Handles data for the Abalone dataset from the UCI Machine Learning repository.
    """
    def __init__(self):
        self.one_hot_vars = ['sex']
        self.response_var = 'rings'
        super().__init__('abalone.csv')

class ServoHandler(DataHandler):
    """
    Handles data for the Servo dataset from the UCI Machine Learning repository.
    """
    def __init__(self):
        self.one_hot_vars = ['motor', 'screw']
        self.response_var = 'rise_time'
        super().__init__('servo.csv')

        
def relu(dim):
    return keras.layers.Dense(
        dim,
        activation=keras.activations.relu,
        kernel_initializer=keras.initializers.HeUniform(),
        bias_initializer=keras.initializers.GlorotUniform())

def make_relu_stack(in_dim, step=2):
    """
    Like make_cycle_stack but just for relus, no cycling.
    """
    start_dim = in_dim - 1
    end_dim = 1
    return [relu(dim) for dim in range(start_dim, end_dim - 1, -step)]

def make_cycle_stack(in_dim,
                     activations: List[Callable],
                     step=2,
                     kernel_initializer=keras.initializers.HeUniform(),
                     bias_initializer=keras.initializers.GlorotUniform()):
    """
    Makes layers for a feed forward NN by starting at `in_dim - 1` 
    and going down in dimension by `step`. Cycles through `activations`
    on the way. For example, if in_dim=10, step=2, activations=[relu, tanh],
    then dense layers of these activation(dimension) would be returned: 
    [relu(9), tanh(7), relu(5), tanh(3), relu(1)] would be returned.
    """
    start_dim = in_dim - 1
    end_dim = 1
    dims = range(start_dim, end_dim - 1, -step)
    layers = []
    for i, dim in enumerate(dims):
        activation_fn = activations[i % len(activations)]
        layer = keras.layers.Dense(
            dim,
            activation=activation_fn,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
        layers.append(layer)
    return layers

def interleave(base_layers: List[Layer], other_layer: Layer) -> List[Layer]:
    new_layers = []
    for layer in base_layers[:-1]:
        new_layers.append(layer)
        new_layers.append(other_layer.copy())
    new_layers.append(base_layers[-1])
    return new_layers

def interleave_dropout(layers: List[Layer], input_dropout=0.8,
                       hidden_dropout=0.5) -> List[Layer]:
    new_layers = []
    if input_dropout > 0.0:
        new_layers.append(keras.layers.Dropout(input_dropout))
    for layer in layers[:-1]:
        new_layers.append(layer)
        if hidden_dropout > 0.0:
            new_layers.append(keras.layers.Dropout(hidden_dropout))
    new_layers.append(layers[-1])
    return new_layers
        
                      
def _prefix_elements(lst, col):
    return [col + '_' + item for item in lst]
