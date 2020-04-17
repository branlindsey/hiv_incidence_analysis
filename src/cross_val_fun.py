import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class XyScaler(BaseEstimator, TransformerMixin):
    """Standardize a training set of data along with a vector of targets."""

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def fit(self, X, y, *args, **kwargs):
        """Fit the scaler to data and a target vector."""
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self
    
    def transform(self, X, y, *args, **kwargs):
        """Transform a new set of data and target vector."""
        return (self.X_scaler.transform(X),
                self.y_scaler.transform(y.reshape(-1, 1)).flatten())

    def inverse_transform(self, X, y, *args, **kwargs):
        """Tranform from a scaled representation back to the original scale."""
        return (self.X_scaler.inverse_transform(X),
                self.y_scaler.inverse_transform(y.reshape(-1, 1)).flatten())
    
def cross_val(X, y, model, n_folds=10, random_seed=154, scale=True):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    scaler = XyScaler()

    for idx, (train_idx, valid_idx) in enumerate (kf.split(X)):
        
        if scale:
            scaler.fit(X[train_idx], y[train_idx])
            std_X_train, std_y_train = scaler.transform(X[train_idx], y[train_idx])
            std_X_valid, std_y_valid = scaler.transform(X[valid_idx], y[valid_idx])
        else:
            std_X_train, std_y_train = X[train_idx], y[train_idx]
            std_X_valid, std_y_valid = X[valid_idx], y[valid_idx]
            
        model.fit(std_X_train,std_y_train)
        y_train_pred = model.predict(std_X_train)
        y_test_pred = model.predict(std_X_valid)
        
        train_cv_errors[idx] = mean_squared_error(std_y_train, y_train_pred)
        test_cv_errors[idx] = mean_squared_error(std_y_valid, y_test_pred)

    return train_cv_errors, test_cv_errors

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    
    cv_errors_train = np.empty(shape=(n_folds, len(alphas)))
    cv_errors_test = np.empty(shape=(n_folds, len(alphas)))
    
    for idx, alpha in enumerate(alphas):
        train_cv_errors, test_cv_errors = cross_val(X, y, model(alpha=alpha), n_folds=n_folds)
        cv_errors_train[:,idx] = train_cv_errors
        cv_errors_test[:,idx] = test_cv_errors
        
    return cv_errors_train, cv_errors_test