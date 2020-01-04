""" Average Individual Fairness learning algorithm based on the paper
`Average Individual Fairness: Algorithms, Generalization and Experiments" by Michael Kearns,
Aaron Roth and Saeed Sharifi-Malvajerdi <https://arxiv.org/abs/1905.10607>_`
Original code from https://github.com/SaeedSharifiMa/AIF/blob/master/ and subsequently adapted.
"""

# documentation:
# implementation of AIF-Learn with a linear threshold oracle


from __future__ import print_function
import functools
import numpy as np
import pandas as pd

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions._regression_learner import RegressionLearner
from fairlearn.reductions._moments.average_individual_fairness import err_rate

print = functools.partial(print, flush=True)


def weighted_predictions(expgrad_result, x):
    """
    Given expgrad_result, compute the weighted predictions over the dataset X
    """
    classifiers = expgrad_result.classifiers
    weights = expgrad_result.weights  # weights over classifiers
    weighted_preds = pd.DataFrame(columns=classifiers.columns)
    for col in classifiers.columns:
        preds = classifiers[col].apply(lambda h: h.predict(x))
        weighted_preds[col] = preds[weights.index].dot(weights)
    return weighted_preds


def _binarize_attribute(Y):
    """
    Binarizes the values for each column.
    """
    if len(Y.shape) == 1 or Y.shape[1] == 1:
        _binarize_column(Y)
        return
    
    for column in Y.columns:
        _binarize_column(Y[column])


def _binarize_column(y):
    if len(np.unique(y)) > 2:  # hack: identify numeric features
        y = 1 * (y > np.mean(y))


class AverageIndividualFairnessLearner:
    """ TODO
    :param alpha: the fairness parameter, called `eps` in ExponentiatedGradient
    :type alpha: float
    """
    def __init__(self, alpha=0.1, nu=0.001, max_iter=1500):
        # Constraints are fixed here and will be created in the fit method.
        # To achieve average individual fairness we need to know n which only happens in fit.
        self._estimator = RegressionLearner()
        self._alpha = alpha
        self._nu = nu
        self._max_iter = max_iter
        self._expgrad = None

    def fit(self, X, y):
        """TODO
        """
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise RuntimeError("AverageIndividualFairnessLearner at this point only supports "
                               "one-dimensional pandas.DataFrames, pandas.Series, one-dimensional"
                               " numpy.ndarrays and lists for the y argument.")

        _binarize_attribute(y)

        n = X.shape[0]

        if len(y.shape) == 1:
            m = 1
        else:
            m = y.shape[1]
        B = 1/self._alpha
        T = max(1, int((B**2) * np.log(n)/self._nu**2))

        constraints = err_rate(range(n))

        self._expgrad = ExponentiatedGradient(self._estimator, constraints=constraints, T=T,
                                              nu=self._nu, eps=self._alpha)
        self._expgrad.fit(X, y, sensitive_features_required=False)

        weighted_preds = weighted_predictions(self._expgrad._expgrad_result, X)

        # error of each problem
        err_problem = {}
        for col in y.columns:
            err_problem[col] = sum(np.abs(y[col] - weighted_preds[col])) / n

        # error of each individual
        err_individual = {}
        for i in range(n):
            err_individual[i] = sum(np.abs(y.loc[i] - weighted_preds.loc[i])) / m

        err_matrix = (y - weighted_preds).abs()
        # overall error rate
        err = err_matrix.values.mean()

        gammahat = self._expgrad._expgrad_result.gammas[self._expgrad._expgrad_result.weights.index].dot(
            self._expgrad._expgrad_result.weights)

        dummy_list = list(err_individual.values())
        for i in range(len(dummy_list)):
            dummy_list[i] = abs(dummy_list[i] - gammahat)
        # unfairness with respect to gammahat
        unf = max(dummy_list)
        # unfairness: maximum disparity between individual error rates
        unf_prime = max(err_individual.values()) - min(err_individual.values())

        # trajectory of error
        error_t = self._expgrad._expgrad_result.error_t
        # trajectory of unfairness
        gamma_t = self._expgrad._expgrad_result.gamma_t

        # set of weights to define the learned mapping, use for new learning problems
        weight_set = self._expgrad._expgrad_result.weight_set

        # weights over classifiers
        weights = self._expgrad._expgrad_result.weights

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
        """TODO
        """
        return weighted_predictions(self._expgrad._expgrad_result, X)
