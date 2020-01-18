# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class ExponentiatedGradientResult:
    """Class to hold the result of an `ExponentiatedGradient` estimator."""

    def __init__(self, best_classifier, best_gap, classifiers, weights, last_t, best_t,
                 n_oracle_calls, gammas, phis, error_t, gamma_t, weight_set):
        self._best_classifier = best_classifier
        self._best_gap = best_gap
        self._classifiers = classifiers
        self._weights = weights
        self._last_t = last_t
        self._best_t = best_t
        self._n_oracle_calls = n_oracle_calls
        self._gammas = gammas
        self._phis = phis
        self._error_t = error_t
        self._gamma_t = gamma_t
        self._weight_set = weight_set

    @property
    def best_classifier(self):
        """Return the best classifier found by the algorithm.

        A function that maps a DataFrame `X` containing covariates to a Series containing the
        corresponding probabilistic decisions in :math:`[0,1]`
        """
        return self._best_classifier

    @property
    def best_gap(self):
        """Return the quality of `best_classifier`.

        If the algorithm has converged then :code:`best_gap <= nu`;
        the solution `best_classifier` is guaranteed to have the classification error within
        :code:`2*best_gap` of the best error under constraint `eps`; the constraint violation
        is at most :code:`2*(eps+best_gap)`
        """
        return self._best_gap

    @property
    def classifiers(self):
        """Return the base classifiers generated (instances of estimator)."""
        return self._classifiers

    @property
    def weights(self):
        """Return the weights of those classifiers within `best_classifier`."""
        return self._weights

    @property
    def last_t(self):
        """Return the last executed iteration; always :code:`last_t < T`."""
        return self._last_t

    @property
    def best_t(self):
        """Return the iteration in which best_classifier was obtained."""
        return self._best_t

    @property
    def n_oracle_calls(self):
        """Return the number of times the estimator was called."""
        return self._n_oracle_calls

    @property
    def gammas(self):
        """TODO"""
        return self._gammas
    
    @property
    def error_t(self):
        """TODO"""
        return self._error_t

    @property
    def gamma_t(self):
        """TODO"""
        return self._gamma_t
    
    @property
    def phis(self):
        """TODO"""
        return self._phis

    @property
    def weight_set(self):
        """TODO"""
        return self._weight_set

    def _as_dict(self):
        return {
            "best_classifier": self._best_classifier,
            "best_gap": self._best_gap,
            "classifiers": self._classifiers,
            "weights": self._weights,
            "last_t": self._last_t,
            "best_t": self._best_t,
            "n_oracle_calls": self._n_oracle_calls,
            "gammas": self._gammas,
            "error_t": self._error_t,
            "gamma_t": self._gamma_t,
            "weight_set": self._weight_set
        }
