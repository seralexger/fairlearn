"""Constraints for individual fairness.
"""

import numpy as np
import pandas as pd

from fairlearn.reductions._regression_learner import RegressionLearner
from fairlearn.reductions._moments.error_rate import ErrorRate


_KW_PHI = "phi"

class err():
    """Misclassification error constraints"""
    short_name = "err"

    def load_data(self, X, Y, **kwargs):
        self.X = X
        self.Y = Y
        self.index = X.index

    def default_objective(self):
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def gamma(self, predictor, **kwargs):
        if _KW_PHI not in kwargs:
            raise RuntimeError("{} is required for gamma.".format(_KW_PHI))
        phi = kwargs[_KW_PHI]

        pred = pd.DataFrame(columns=predictor.columns)
        for column in predictor.columns:
            pred.loc[0, column] = predictor.loc[0, column].predict(self.X)
        pred = pred.loc[0]  # convert to series
        g_unsigned = pd.Series(data=(self.Y-pred).abs().mean() - phi, index=self.index)
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=["sign", "constraint"])
        return g_signed

    def signed_weights(self, lambda_vec):
        return lambda_vec["+"] - lambda_vec["-"]


class err_named(err):

    short_name = "err_named"

    def __init__(self, constraint):
        self.constraint = constraint

    def load_data(self, X, Y, **kwargs):
        super().load_data(X.loc[[self.constraint]], Y.loc[self.constraint])


class err_rate():

    short_name = "err_rate"

    def __init__(self, constraints):
        self.constraints = constraints
        self.err = {}
        for i in self.constraints:
            self.err[i] = err_named(i)

    def load_data(self, X, Y, **kwargs):
        # need to access Y's column names so convert to DataFrame if it's a Series
        if type(Y) == pd.Series:
            Y_df = Y.to_frame()
        else:
            Y_df = Y

        for i in self.constraints:
            self.err[i].load_data(X, Y_df)
        dummy_predictor = pd.DataFrame(columns=Y_df.columns)
        dummy_weights = pd.Series(0, index=np.arange(len(X)))
        for column in Y_df.columns:
            dummy_predictor.loc[0, column] = RegressionLearner()
            dummy_predictor.loc[0, column].fit(X=X, Y=Y_df[Y_df.columns[0]], sample_weight=dummy_weights)
        dummy_gamma = self.gamma(dummy_predictor, phi=1)
        self.index = dummy_gamma.index
    
    def default_objective(self):
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def gamma(self, predictor, **kwargs):
        if _KW_PHI not in kwargs:
            raise RuntimeError("{} is required for gamma.".format(_KW_PHI))
        phi = kwargs[_KW_PHI]

        gamma = {}
        for i in self.constraints:
            gamma[i] = self.err[i].gamma(predictor, phi=phi)
        return pd.concat(list(gamma.values()), keys=self.constraints)

    def signed_weights(self, lambda_vec):
        lambda_dict = {}
        for i in self.constraints:
            lambda_dict[i] = self.err[i].signed_weights(lambda_vec[i])
        return pd.concat(list(lambda_dict.values()), keys=self.constraints)
    
    def project_lambda(self, lambda_vec):
        return self.signed_weights(lambda_vec)
