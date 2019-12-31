""" Average Individual Fairness learning algorithm based on the paper
`Average Individual Fairness: Algorithms, Generalization and Experiments" by Michael Kearns,
Aaron Roth and Saeed Sharifi-Malvajerdi <https://arxiv.org/abs/1905.10607>_`
Original code from https://github.com/SaeedSharifiMa/AIF/blob/master/ and subsequently adapted.
"""

# documentation:
# implementation of AIF-Learn with a linear threshold oracle


from __future__ import print_function
import argparse
import functools
import numpy as np
import pandas as pd
import clean_data as parser1
import pickle
from sklearn import linear_model
from scipy.optimize import fsolve

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions._moments.average_individual_fairness import err_rate

print = functools.partial(print, flush=True)


class RegressionLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, W):
        cost_vec0 = Y * W  # cost vector for predicting zero
        cost_vec1 = (1 - Y) * W  # cost vector for predicting one
        self.reg0 = linear_model.LinearRegression()
        self.reg0.fit(X, cost_vec0)
        self.reg1 = linear_model.LinearRegression()
        self.reg1.fit(X, cost_vec1)

    def predict(self, X):
        pred0 = self.reg0.predict(X)
        pred1 = self.reg1.predict(X)
        return 1*(pred1 < pred0)


def weighted_predictions(expgrad_result, x):
    """
    Given expgrad_result, compute the weighted predictions
    over the dataset x
    """
    classifiers = expgrad_result.classifiers
    weights = expgrad_result.weights  # weights over classifiers
    weighted_preds = pd.DataFrame(columns=classifiers.columns)
    for col in classifiers.columns:
        preds = classifiers[col].apply(lambda h: h.predict(x))
        weighted_preds[col] = preds[weights.index].dot(weights)
    return weighted_preds


def _binarize_attribute(a):
    """
    given a set of attributes; binarize them
    """
    for col in a.columns:
        if len(a[col].unique()) > 2:  # hack: identify numeric features
            sens_mean = np.mean(a[col])
            a[col] = 1 * (a[col] > sens_mean)


class AverageIndividualFairnessLearner:
    """

    :param alpha: the fairness parameter
    """

    def __init__(self, alpha=0.1, nu=0.001, max_iter=1500):
        self._alpha = alpha
        self._nu = nu
        self._max_iter = max_iter

    def fit(self, X, y):
        # TODO Remove this dataset specific code
        """
        if dataset == 'communities':
            x, y = parser1.clean_communities()
        elif dataset == 'synthetic':
            x, y = parser1.clean_synthetic()
        else:
            raise Exception('Dataset not in range!')
        """
        print(x.columns)
        print(y.columns)

        _binarize_attribute(y)

        ############ setting some parameters ############
        n = X.shape[0]
        m = y.shape[1]
        B = 1/self._alpha
        T_numerator = (B**2) * np.log(n)
        T_denominator = self._nu**2
        T = T_numerator/T_denominator
        T = max(1, int(T))
        #################################################

        ############### Define the learning oracle and the constraints ###################
        learner = RegressionLearner()
        constraints = err_rate(range(n))
        ##################################################################################

        ################ Run the algorithm ##################
        expgrad = ExponentiatedGradient(learner, constraints=constraints, T=T, nu=self._nu)
        expgrad.fit(X, y)
        #####################################################

        weighted_preds = weighted_predictions(expgrad._expgrad_result, X)

        err_problem = {}  # error of each problem
        for col in y.columns:
            err_problem[col] = sum(np.abs(y[col] - weighted_preds[col])) / n

        err_individual = {}  # error of each individual
        for i in range(n):
            err_individual[i] = sum(np.abs(y.loc[i] - weighted_preds.loc[i])) / m

        err_matrix = (y - weighted_preds).abs()
        err = err_matrix.values.mean()  # overall error rate

        gammahat = expgrad._expgrad_result.gammas[expgrad._expgrad_result.weights.index].dot(
            expgrad._expgrad_result.weights)

        dummy_list = list(err_individual.values())
        for i in range(len(dummy_list)):
            dummy_list[i] = abs(dummy_list[i] - gammahat)
        unf = max(dummy_list)  # unfairness with respect to gammahat
        # unfairness: maximum disparity between individual error rates
        unf_prime = max(err_individual.values()) - min(err_individual.values())

        error_t = expgrad._expgrad_result.error_t  # trajectory of error
        gamma_t = expgrad._expgrad_result.gamma_t  # trajectory of unfairness

        # set of weights to define the learned mapping, use for new learning problems
        weight_set = expgrad._expgrad_result.weight_set

        weights = expgrad._expgrad_result.weights  # weights over classifiers

        # TODO create result object
        d = {'err_matrix': err_matrix,
             'err': err,
             'unfairness': unf,
             'unfairness_prime': unf_prime,
             'alpha': self._alpha,
             'err_problem': err_problem,
             'err_individual': err_individual,
             'gammahat': gammahat,
             'error_t': error_t,
             'gamma_t': gamma_t,
             'weight_set': weight_set,
             'weights': weights,
             'individuals': X}
        d_print = {'err': err,
                   'gammahat': gammahat,
                   'unfairness': unf,
                   'unfairness_prime': unf_prime,
                   'alpha': self._alpha}
        print(d_print)

    def predict(self, X):
        pass
        # TODO: figure out how to do predictions
