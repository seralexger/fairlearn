# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt

from ._constants import _PRECISION, _INDENTATION, _LINE

from fairlearn.reductions._moments.average_individual_fairness import err_rate

logger = logging.getLogger(__name__)


class _Lagrangian:
    """Operations related to the Lagrangian.

    :param X: the training features
    :type X: Array
    :param sensitive_features: the sensitive features to use for constraints
    :type sensitive_features: Array
    :param y: the training labels
    :type y: Array
    :param estimator: the estimator to fit in every iteration of best_h
    :type estimator: an estimator that has a `fit` method with arguments X, y, and sample_weight
    :param constraints: Object describing the parity constraints. This provides the reweighting
        and relabelling
    :type constraints: `fairlearn.reductions.Moment`
    :param eps: allowed constraint violation
    :type eps: float
    :param B:
    :type B:
    :param opt_lambda: optional with default value True
    :type opt_lambda: bool
    """

    def __init__(self, X, sensitive_features, y, estimator, constraints, eps, B, opt_lambda=True, eval_gap=True):
        self.X = X
        self.constraints = constraints
        self.constraints.load_data(X, y, sensitive_features=sensitive_features)
        self.obj = self.constraints.default_objective()
        self.obj.load_data(X, y, sensitive_features=sensitive_features)
        self.pickled_estimator = pickle.dumps(estimator)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self._eval_gap = eval_gap
        self.hs = pd.Series()
        self.classifiers = pd.Series()
        self.errors = pd.Series()
        self.gammas = pd.DataFrame()
        self.phis = pd.Series()  # these actually correspond to gamma_t in the paper
        self.lambdas = pd.DataFrame()
        self.n = self.X.shape[0]
        self.n_oracle_calls = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None
        self.weight_set = pd.DataFrame()

        # TODO remove this weird hack to get column names
        if type(y) == pd.DataFrame:
            self.labels = y.columns
        elif type(y) == pd.Series:
            self.labels = y.to_frame().columns
        else:
            self.labels = pd.DataFrame(y).columns
        
        self.y = y

    def _eval_from_error_gamma(self, error, gamma, lambda_vec):
        """Return the value of the Lagrangian.

        :return: tuple `(L, L_high)` where `L` is the value of the Lagrangian, and
            `L_high` is the value of the Lagrangian under the best response of the lambda player
        :rtype: tuple of two floats
        """
        lambda_projected = self.constraints.project_lambda(lambda_vec)
        if self.opt_lambda:
            L = error + np.sum(lambda_projected * gamma) - self.eps * np.sum(lambda_projected)
        else:
            L = error + np.sum(lambda_vec * gamma) - self.eps * np.sum(lambda_vec)
        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B * (max_gamma - self.eps)
        return L, L_high

    def _eval(self, h, lambda_vec):
        """Return the value of the Lagrangian.

        :return: tuple `(L, L_high, gamma, error)` where `L` is the value of the Lagrangian,
            `L_high` is the value of the Lagrangian under the best response of the lambda player,
            `gamma` is the vector of constraint violations, and `error` is the empirical error
        """
        # TODO changes required here?
        print()
        print('h {}'.format(h))
        print()

        if callable(h):
            error = self.obj.gamma(h)[0]
            gamma = self.constraints.gamma(h)
        else:
            error = self.errors[h.index].dot(h)
            gamma = self.gammas[h.index].dot(h)
        
        print()
        print('error {}'.format(error))
        print()
        print()
        print('gamma {}'.format(gamma))
        print()
        L, L_high = self._eval_from_error_gamma(error, gamma, lambda_vec)
        return L, L_high, gamma, error

    def eval_gap(self, h, lambda_hat, nu):
        r"""Return the duality gap object for the given :math:`h` and :math:`\hat{\lambda}`."""
        L, L_high, gamma, error = self._eval(h, lambda_hat)
        result = _GapResult(L, L, L_high, gamma, error)
        if self._eval_gap:
            for mul in [1.0, 2.0, 5.0, 10.0]:
                h_hat, h_hat_idx = self.best_h(mul * lambda_hat)
                logger.debug("%smul=%.0f", _INDENTATION, mul)
                L_low_mul, _, _, _ = self._eval(
                    pd.Series({h_hat_idx: 1.0}), lambda_hat)
                if L_low_mul < result.L_low:
                    result.L_low = L_low_mul
                if result.gap() > nu + _PRECISION:
                    break
        return result

    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_constraints = len(self.constraints.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_result
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate((self.gammas - self.eps, -np.ones((n_constraints, 1))), axis=1)
        b_ub = np.zeros(n_constraints)
        A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
        h = pd.Series(result.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate((-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in range(n_constraints + 1)]  # noqa: E501
        result_dual = opt.linprog(dual_c,
                                  A_ub=dual_A_ub,
                                  b_ub=dual_b_ub,
                                  bounds=dual_bounds,
                                  method='simplex')
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (h, lambda_vec, self.eval_gap(h, lambda_vec, nu))
        return self.last_linprog_result

    def best_h(self, lambda_vec):
        """Solve the best-response problem.

        Returns the classifier that solves the best-response problem for
        the vector of Lagrange multipliers `lambda_vec`.
        """
        # TODO if we want to work with multiple classification problems then this needs some change
        lambda_signed = self.constraints.signed_weights(lambda_vec)
        print()
        print("lambda_signed {}".format(lambda_signed))
        print()
        lambda_signed.index = lambda_signed.index.droplevel(1)

        if type(self.constraints) is err_rate:
            redW = lambda_signed + 1/self.n
            redY = self.y
        else:
            signed_weights = self.obj.signed_weights() + lambda_signed
            redY = 1 * (signed_weights > 0)
            signed_weights_abs = signed_weights.abs()
            redW = self.n * signed_weights_abs / signed_weights_abs.sum()
        
        phi = 1*(np.sum(lambda_signed) > 0)
        print()
        print("redW {}".format(redW))
        print()
        print("phi {}".format(phi))
        print()

        classifier = pickle.loads(self.pickled_estimator)
        classifier.fit(self.X, redY, sample_weight=redW)
        self.n_oracle_calls += 1

        def h(X): return classifier.predict(X)
        h_error = self.obj.gamma(h)[0]

        # AIF works with multiple classification problems and therefore expects multiple
        # classifiers. This is a hack and should be done properly.
        if type(self.constraints) is err_rate:
            classifiers = pd.DataFrame(columns=self.labels)
            classifiers.loc[0, self.labels[0]] = classifier
            h_gamma = self.constraints.gamma(classifiers, phi=phi)
        else:
            h_gamma = self.constraints.gamma(h)
        h_value = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            values = self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = values.idxmin()
            best_value = values[best_idx]
        else:
            best_idx = -1
            best_value = np.PINF

        if h_value < best_value - _PRECISION:
            logger.debug("%sbest_h: val improvement %f", _LINE, best_value - h_value)
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.weight_set[h_idx] = redW
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            self.lambdas[h_idx] = lambda_vec.copy()
            self.phis.at[h_idx] = phi
            best_idx = h_idx

        return self.hs[best_idx], best_idx


class _GapResult:
    """The result of a duality gap computation."""

    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L - self.L_low, self.L_high - self.L)
