# -*- coding: utf-8 -*-
"""
Created on Tue May 14 2019
@author: Hideyuki Nakano
"""
import numpy as np
from sklearn.utils import resample
import pandas as pd

def uncertain( data, name_s, num_exp, num_t, n_bootstraps ):
    """
    This function is based on the error estimation by
    Wakamatsu et al 2017, Hydrological Research Letter 11(1), 65-71
    Input:
        data:   2d data [num_exp, num_t]
        name_s: name of secenario
        num_t  : number of timeseries for each experiment
        num_exp: number of experiments for the scenario
        n_bootstraps: number of bootstraps
    Output:
        pd.Series[mean, std, M, V, B], name=name_exp, index=['mean', 'std', 'M', 'V', 'B'])
        mean: ensemble mean of all models
        std : sqrt(M+V+B)
        M:  model uncertainity
        V:  internal variability
        B:  Bootstrap term (uncertainty due to sample size)
    Examples
    --------
      >>> sr = uncertain( hist, 'Present', nexp, nt, num_bootstraps )
    """
    mu = np.mean(data)                                 # eq(1)
    alpha = np.mean(data, axis=1, keepdims=True) - mu  # eq(1)
    epsil = data - mu - alpha                          # eq(1)
    M = np.var(alpha)                                  # eq(2)
    V = np.var(epsil, ddof = num_exp)                  # eq(3)

    data_flat = data.flatten()
    mu_bootstrap = []
    for _ in range(n_bootstraps): 
        mu_samples = resample(data_flat, n_samples = num_t )
        mu_bootstrap.append(np.mean(mu_samples))
    B = np.var(mu_bootstrap,ddof=1)    # eq (12)
    std = np.sqrt(M+V+B)
    return pd.Series([mu, std, M, V, B ], name=name_s, index=['mean', 'std', 'M', 'V', 'B'])

def uncertain_2d( data, n_bootstraps ):
    """
    This function is based on the error estimation by
    Wakamatsu et al 2017, Hydrological Research Letter 11(1), 65-71
    Input:
        data:   4d data [num_exp, num_t, ny, nx]
        n_bootstraps: number of bootstraps
    Output:
        numpy.array([mean, std, M, V, B])
        mean: ensemble mean of all models
        std : sqrt(M+V+B)
        M:  model uncertainity
        V:  internal variability
        B:  Bootstrap term (uncertainty due to sample size)
    Examples
    --------
      >>> result = uncertain_2d( data, num_bootstraps )
    """
    num_exp, num_t, ny, nx = data.shape
    
    mu = np.mean(data,axis=(0,1))                      # eq(1)
    alpha = np.mean(data, axis=1, keepdims=True) - mu  # eq(1)
    epsil = data - mu - alpha                          # eq(1)
    M = np.var(alpha,axis=(0,1))                       # eq(2)
    V = np.var(epsil,axis=(0,1), ddof = num_exp)       # eq(3)
    data_flat = data.reshape(num_exp*num_t,ny,nx)
    mu_bootstrap = []
    for _ in range(n_bootstraps):
        mu_samples = resample( data_flat, n_samples = num_t )
        mu_bootstrap.append( np.mean(mu_samples,axis=0) )
    B = np.var(mu_bootstrap,axis=0,ddof=1)    # eq (12)
    std = np.sqrt(M+V+B)
    return np.array([ mu, std, M, V, B ])
