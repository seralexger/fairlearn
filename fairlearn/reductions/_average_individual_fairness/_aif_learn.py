# documentation:
# implementation of AIF-Learn with a linear threshold oracle


from __future__ import print_function
import argparse
import functools
import numpy as np
import pandas as pd
import classred as red
import clean_data as parser1
import pickle
from sklearn import linear_model
from scipy.optimize import fsolve

print = functools.partial(print, flush=True)


############################### LEARNING ORACLE ##############################
class RegressionLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, W):
        cost_vec0 = Y * W  # cost vector for predicting zero
        cost_vec1 = (1 - Y) * W # cost vector for predicting one
        self.reg0 = linear_model.LinearRegression()
        self.reg0.fit(X, cost_vec0)
        self.reg1 = linear_model.LinearRegression()
        self.reg1.fit(X, cost_vec1)
      
    def predict(self, X):
        pred0 = self.reg0.predict(X)
        pred1 = self.reg1.predict(X)
        return 1*(pred1 < pred0)
##############################################################################


############################# AIF CONSTRAINTS ################################
class err():
    """Misclassification error constraints"""
    short_name = "err"

    def init(self, dataX, dataY):
        self.X = dataX
        self.Y = dataY
        self.index = dataX.index

    def gamma(self, predictor, phi):
        pred = pd.DataFrame(columns = predictor.columns)
        for col in predictor.columns:
            pred.loc[0, col] = predictor.loc[0, col].predict(self.X)
        pred = pred.loc[0] # convert to series
        g_unsigned = pd.Series(data = (self.Y-pred).abs().mean() - phi, index = self.index)
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+","-"],
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
        dummy_predictor = pd.DataFrame(columns = dataY.columns)
        dummy_weights = pd.Series(0, index=np.arange(len(dataX)))
        for col in dataY.columns:
            dummy_predictor.loc[0,col] = RegressionLearner()
            dummy_predictor.loc[0,col].fit(X = dataX, Y = dataY[dataY.columns[0]], W = dummy_weights)
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
################################################################################



def weighted_predictions(res_tuple, x):
    """
    Given res_tuple from expgrad, compute the weighted predictions
    over the dataset x
    """
    classifiers = res_tuple.classifiers
    weights = res_tuple.weights  # weights over classifiers
    weighted_preds = pd.DataFrame(columns = classifiers.columns)
    for col in classifiers.columns:
        preds = classifiers[col].apply(lambda h: h.predict(x))
        weighted_preds[col] =  preds[weights.index].dot(weights)
    return weighted_preds


def binarize_attr(a):
    """
    given a set of attributes; binarize them
    """
    for col in a.columns:
        if len(a[col].unique()) > 2:  # hack: identify numeric features
            sens_mean = np.mean(a[col])
            a[col] = 1 * (a[col] > sens_mean)


def run_aif_learn(dataset, alpha = 0.1, nu = 0.001, max_iter = 1500):

    if dataset == 'communities':
        x, y = parser1.clean_communities()
    elif dataset == 'synthetic':
        x, y = parser1.clean_synthetic()
    else:
        raise Exception('Dataset not in range!')

    print(x.columns)
    print(y.columns)

    binarize_attr(y)
    
    ############ setting some parameters ############
    n = x.shape[0]
    m = y.shape[1]
    B = 1/alpha
    T_numerator = (B**2) * np.log(n)
    T_denominator = nu**2
    T = T_numerator/T_denominator
    T = max(1, int(T))
    eta = nu / B
    #################################################
    
    ############### Define the learning oracle and the constraints ###################
    learner = RegressionLearner()
    constraints = err_rate(range(n))
    ##################################################################################
    
    ################ Run the algorithm ##################
    res_tuple = red.expgrad(x, y, learner, cons=constraints, alpha = alpha, 
                            B=B, T=T, nu = nu, debug=True, max_iter=max_iter)          
    #####################################################


    weighted_preds = weighted_predictions(res_tuple, x)
    
    err_problem = {} # error of each problem
    for col in y.columns:
        err_problem[col] = sum(np.abs(y[col] - weighted_preds[col])) / n
    
    err_individual = {} # error of each individual
    for i in range(n):
        err_individual[i] = sum(np.abs(y.loc[i] - weighted_preds.loc[i])) / m
    
    err_matrix = (y - weighted_preds).abs()
    err = err_matrix.values.mean() # overall error rate
    
    gammahat = res_tuple.phis[res_tuple.weights.index].dot(res_tuple.weights) # gammahat
    
    dummy_list = list(err_individual.values())
    for i in range(len(dummy_list)):
        dummy_list[i] = abs(dummy_list[i] - gammahat)
    unf = max(dummy_list) # unfairness with respect to gammahat
    unf_prime = max(err_individual.values()) - min(err_individual.values()) # unfairness: maximum disparity between individual error rates

    error_t = res_tuple.error_t # trajectory of error
    gamma_t = res_tuple.gamma_t # trajectory of unfairness

    weight_set = res_tuple.weight_set # set of weights to define the learned mapping, use for new learning problems
    
    weights = res_tuple.weights  # weights over classifiers

    d = {'err_matrix': err_matrix, 'err' :err , 'unfairness' : unf, 'unfairness_prime': unf_prime, 'alpha' : alpha, 'err_problem': err_problem , 
            'err_individual': err_individual, 'gammahat': gammahat, 'error_t': error_t, 'gamma_t':gamma_t,
            'weight_set': weight_set, 'weights': weights, 'individuals': x}
    d_print = {'err' :err , 'gammahat': gammahat, 'unfairness' : unf, 'unfairness_prime': unf_prime, 'alpha' : alpha}    
    print(d_print)
    return d








data_list = ['communities', 'synthetic'] # accepted data sets, add as needed


def setup_argparse():
  parser = argparse.ArgumentParser('run AIF learn')
  parser.add_argument('-nu', '--nu', type=float, default=0.001, help='the approximation parameter input to the algorithm')
  parser.add_argument('-alpha', '--alpha', type=float, default=0.1, help='fairness parameter')
  parser.add_argument('-d', '--dataset', choices=data_list, help='dataset to analyse')
  parser.add_argument('-max_iter', '--max_iter', type=int, default=1500, help='max iteration of the algorithm')
  return parser



if __name__=='__main__':
  parser = setup_argparse()
  args = parser.parse_args()
  data = run_aif_learn(args.dataset, args.alpha, args.nu, args.max_iter)
  pickle.dump(data, open('pickles/{}_{}_{}_{}_aif.p'.format(args.dataset, args.max_iter, args.alpha, args.nu), 'wb'))
