# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from fairlearn.reductions import EqualizedOdds, DemographicParity, err_rate
from constants import ADULT_UCI, COMPAS, COMMUNITIES_UCI, RBM_SVM, DECISION_TREE_CLASSIFIER, \
    THRESHOLD_OPTIMIZER, EXPONENTIATED_GRADIENT, GRID_SEARCH, AVERAGE_INDIVIDUAL_FAIRNESS_LEARNER

try:
    from tempeh.execution.azureml.workspace import get_workspace
except ImportError:
    raise Exception("fairlearn performance tests require azureml-sdk to be installed.")

from tempeh import configurations as tempeh_conf
from tempeh import constants as tempeh_const

from environment_setup import build_package


DATASETS = [
    ADULT_UCI,
    COMPAS,
    COMMUNITIES_UCI
]
PREDICTORS = [
    RBM_SVM,
    DECISION_TREE_CLASSIFIER
]
MITIGATORS = [
    THRESHOLD_OPTIMIZER,
    EXPONENTIATED_GRADIENT,
    GRID_SEARCH,
    AVERAGE_INDIVIDUAL_FAIRNESS_LEARNER
]


class PerfTestConfiguration:
    def __init__(self, dataset, predictor, mitigator, disparity_metric):
        self.dataset = dataset
        self.predictor = predictor
        self.mitigator = mitigator
        self.disparity_metric = disparity_metric

    def __repr__(self):
        return "[dataset: {}, predictor: {}, mitigator: {}, disparity_metric: {}]" \
               .format(self.dataset, self.predictor, self.mitigator, self.disparity_metric)


def get_all_perf_test_configurations():
    perf_test_configurations = []
    for dataset in DATASETS:
        dataset_class = tempeh_conf.datasets[dataset]
        for predictor in PREDICTORS:
            predictor_class = tempeh_conf.models[predictor]
            if not predictor_class.compatible_with_dataset(dataset_class):
                print("Predictor {} is not compatible with dataset {}. Skipping..."
                      .format(predictor, dataset))
                continue

            # TODO: encode mitigator compatibility somewhere
            for mitigator in MITIGATORS:
                if mitigator == THRESHOLD_OPTIMIZER:
                    if dataset_class.task != tempeh_const.Tasks.BINARY:
                        print("Mitigator {} is not compatible with dataset {}. Skipping..."
                              .format(mitigator, dataset))
                        continue

                    disparity_metrics = ["equalized_odds", "demographic_parity"]

                elif mitigator == EXPONENTIATED_GRADIENT:
                    if dataset_class.task != tempeh_const.Tasks.BINARY:
                        print("Mitigator {} is not compatible with dataset {}. Skipping..."
                              .format(mitigator, dataset))
                        continue

                    disparity_metrics = [EqualizedOdds.__name__, DemographicParity.__name__]

                elif mitigator == GRID_SEARCH:
                    if dataset_class.task != tempeh_const.Tasks.BINARY:
                        print("Mitigator {} is not compatible with dataset {}. Skipping..."
                              .format(mitigator, dataset))
                        continue

                    disparity_metrics = [EqualizedOdds.__name__, DemographicParity.__name__]

                elif mitigator == AVERAGE_INDIVIDUAL_FAIRNESS_LEARNER:
                    if dataset_class.task != tempeh_const.Tasks.REGRESSION:
                        print("Mitigator {} is not compatible with dataset {}. Skipping..."
                              .format(mitigator, dataset))
                        continue

                    # We're not providing this explicitly since it's the only one that currently
                    # works with AIF Learner, so this is just a dummy value.
                    disparity_metrics = [err_rate.__name__]

                else:
                    raise Exception("Unknown mitigator {}".format(mitigator))

                for disparity_metric in disparity_metrics:
                    perf_test_configurations.append(
                        PerfTestConfiguration(dataset, predictor, mitigator, disparity_metric))

    return perf_test_configurations


@pytest.fixture(scope="session")
def workspace():
    return get_workspace()


@pytest.fixture(scope="session")
def wheel_file():
    return build_package()
