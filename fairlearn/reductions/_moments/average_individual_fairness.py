"""Constraints for individual fairness.
"""

import numpy as np
import pandas as pd

from fairlearn.reductions._regression_learner import RegressionLearner


class err():
    """Misclassification error constraints"""
    short_name = "err"

    def init(self, dataX, dataY):
        self.X = dataX
        self.Y = dataY
        self.index = dataX.index

    def gamma(self, predictor, phi):
        pred = pd.DataFrame(columns=predictor.columns)
        for col in predictor.columns:
            pred.loc[0, col] = predictor.loc[0, col].predict(self.X)
        pred = pred.loc[0]  # convert to series
        g_unsigned = pd.Series(data=(self.Y-pred).abs().mean() - phi, index=self.index)
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=["sign", "constraint"])
        return g_signed

    def lambda_signed(self, lambda_vec):
        return lambda_vec["+"] - lambda_vec["-"]


class err_named(err):

    short_name = "err_named"

    def __init__(self, constraint):
        self.constraint = constraint

    def init(self, dataX, dataY):
        super().init(dataX.loc[[self.constraint]], dataY.loc[self.constraint])


class err_rate():

    short_name = "err_rate"

    def __init__(self, constraints):
        self.constraints = constraints
        self.err = {}
        for i in self.constraints:
            self.err[i] = err_named(i)

    def init(self, dataX, dataY):
        for i in self.constraints:
            self.err[i].init(dataX, dataY)
        dummy_predictor = pd.DataFrame(columns=dataY.columns)
        dummy_weights = pd.Series(0, index=np.arange(len(dataX)))
        for col in dataY.columns:
            dummy_predictor.loc[0, col] = RegressionLearner()
            dummy_predictor.loc[0, col].fit(X=dataX, Y=dataY[dataY.columns[0]], W=dummy_weights)
        dummy_gamma = self.gamma(dummy_predictor, 1)
        self.index = dummy_gamma.index

    def gamma(self, predictor, phi):
        gamma = {}
        for i in self.constraints:
            gamma[i] = self.err[i].gamma(predictor, phi)
        return pd.concat(list(gamma.values()), keys=self.constraints)

    def lambda_signed(self, lambda_vec):
        lambda_dict = {}
        for i in self.constraints:
            lambda_dict[i] = self.err[i].lambda_signed(lambda_vec[i])
        return pd.concat(list(lambda_dict.values()), keys=self.constraints)
