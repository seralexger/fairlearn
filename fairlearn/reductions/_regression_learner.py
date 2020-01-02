# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sklearn import linear_model


class RegressionLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, sample_weight):
        cost_vec0 = Y * sample_weight  # cost vector for predicting zero
        cost_vec1 = (1 - Y) * sample_weight  # cost vector for predicting one
        self.reg0 = linear_model.LinearRegression()
        self.reg0.fit(X, cost_vec0)
        self.reg1 = linear_model.LinearRegression()
        self.reg1.fit(X, cost_vec1)

    def predict(self, X):
        pred0 = self.reg0.predict(X)
        pred1 = self.reg1.predict(X)
        return 1*(pred1 < pred0)
