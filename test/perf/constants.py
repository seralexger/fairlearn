# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch, \
    AverageIndividualFairnessLearner


THRESHOLD_OPTIMIZER = ThresholdOptimizer.__name__
EXPONENTIATED_GRADIENT = ExponentiatedGradient.__name__
GRID_SEARCH = GridSearch.__name__
AVERAGE_INDIVIDUAL_FAIRNESS_LEARNER = AverageIndividualFairnessLearner.__name__

MEMORY = "memory"
TIME = "time"

ADULT_UCI = 'adult_uci'
COMPAS = 'compas'
COMMUNITIES_UCI = "communities_uci"

RBM_SVM = 'rbm_svm'
DECISION_TREE_CLASSIFIER = 'decision_tree_classifier'
