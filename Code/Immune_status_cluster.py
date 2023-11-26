#!/usr/bin/env python
# encoding: utf-8

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
sns.set(style='ticks')

# Read the dataset
data=pd.read_csv('CBC_log_norm.csv',encoding='gbk')  # Read the data

# Check for missing values in the data
print(data.isnull().any())   # Check for missing values in each "column"
print(data[data.isnull().T.any().T])  # Find all rows containing nan
data.dropna(axis=0, how='any', inplace=True)
print('Rows:',data.shape[0],',  Columns:',data.shape[1]) # Check the number of rows and columns
data.head(5)
data.describe()
data = data.iloc[:,1:16]

# EM Algorithm for GMM Model Prediction with 3 Classes
plt.rcParams['font.family'] = 'Arial Unicode MS'  # Chinese support
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Debug function to control output based on the global variable DEBUG
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args,**kwargs)

# Calculate the probability density function for the k-th model
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

# E-step: Calculate the responsiveness of each model to the samples
def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]  # Number of samples
    K = alpha.shape[0]  # Number of models

    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    gamma = np.mat(np.zeros((N, K)))  # Responsiveness matrix
    prob = np.zeros((N, K))  # Probability matrix for each model and sample
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

# M-step: Iterate model parameters
def maximize(Y, gamma):
    N, D = Y.shape  # Number of samples and features
    K = gamma.shape[1]  # Number of models

    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    for k in range(K):
        Nk = np.nansum(gamma[:, k])  # Sum of responsiveness for the k-th model
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha

# Scale the data to be between 0 and 1
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y

# Initialize model parameters
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha

# Gaussian Mixture Model EM Algorithm
def GMM_EM(Y, K, times):
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha

# Read the dataset and apply GMM EM algorithm
Y = data
matY = np.matrix(Y, copy=True)
K = 3  # Number of models
mu, cov, alpha = GMM_EM(matY, K, 100)
# Calculate the responsiveness matrix for the current model parameters
gamma = getExpectation(matY, mu, cov, alpha)
category = gamma.argmax(axis=1).flatten().tolist()[0]  # Find the model index with the maximum responsiveness for each sample


