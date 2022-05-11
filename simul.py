from scipy.stats import expon, chi2, genextreme, truncnorm, multivariate_normal, norm

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

from cvxopt import matrix, solvers
from utils import *

solvers.options['show_progress']=False

def visualize_quantile(portfolio, gamma, beta, delta):
    """ Visualizing portfoilo quantile function 
    
    Estimated quantile function vs Empirical quantile function

    Args:
        portfolio ([tensor]): [description]
        gamma ([tensor]): [description]
        beta ([tensor]): [description]
        delta ([tensor]): [description]
        
    """
    mpl.style.use('seaborn')
    quantile_ = np.linspace(0, 1, 100, dtype=np.float32)
    estimated_return = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.constant(quantile_)))
    empirical_quantile = np.quantile(np.squeeze(portfolio.numpy()), [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])
    plt.plot(np.linspace(0, 1, 100, dtype=np.float32), estimated_return.numpy(), label='Estimated Quantile')
    plt.plot(np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1], dtype=np.float32), empirical_quantile, label='Empirical Quantile')
    plt.xlabel("portfolio return")
    plt.ylabel("quantile")
    plt.legend()
    plt.title("Estimated Portfolio Quantile")
    plt.show()
    
def param_distribution_upr(dist_type, size, seed, epoch):
    """ 
    Sampling from parametric distribution which is defined by dist_type argument

    Args:
        dist_type ([str]): Generating distribution, Exponential: 'ex', Chi-squared: 'chi', Truncated normal: 'tn', GEV: 'gev'
        size ([int]): Sample size
        seed ([int]): Seed number
        epoch ([int]): Iteration epochs
    
    Returns:
        gamma ([tensor]): gamma of linear isotonic regression spline 
        beta ([tensor]): beta of linear isotonic regression spline
    """
    if dist_type == 'ex':
        sample = tf.constant(expon.rvs(loc=0, scale=0.5, size=size, random_state=seed), dtype=tf.float32)

    elif dist_type == 'chi':
        sample =  tf.constant(chi2.rvs(2, size=size, random_state=seed), dtype=tf.float32)
    
    elif dist_type == 'tn':
        sample = tf.constant(truncnorm.rvs(a=-2.5, b=0.5, loc=2.5, scale=1, size=size, random_state=seed), dtype=tf.float32)    

    elif dist_type == 'gev1':
        sample = tf.constant(genextreme.rvs(loc=2, c=-0.3, size=size, random_state=seed), dtype=tf.float32)
        
    else:
        sample = tf.constant(genextreme.rvs(loc=2, c=0.5, size=size, random_state=seed), dtype=tf.float32)

    beta = tf.Variable(tf.random.uniform([14], minval=0.0, maxval=0.1, dtype=tf.dtypes.float32)[tf.newaxis,:])

    delta = tf.constant(np.append(np.append([0.0, 0.05, 0.05], np.repeat(1, 8)/10), [0.05, 0.05, 0.0]), dtype=tf.dtypes.float32)

    gamma = tf.Variable(tf.constant(np.array([0.0]), dtype=tf.dtypes.float32))

    initial_learning_rate1 = 0.01
    
    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate1,
    decay_steps=epoch,
    decay_rate=0.97,
    staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule1)    
    
    NUM_ITER = tqdm(range(epoch))
    
    for j in NUM_ITER:
        with tf.GradientTape() as tape:
            quantile_dl = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.math.cumsum(delta)))
            
            upr_  = tf.map_fn(lambda x: upr(x, gamma, beta, delta, quantile_dl), sample)
                        
            loss = tf.reduce_mean(upr_)
            #loss = tf.reduce_sum(upr_)
            
        if (epoch+1) % 10 == 0:
            NUM_ITER.set_postfix({'loss': loss.numpy()})
                    
        grads = tape.gradient(loss, [gamma, beta])
        optimizer.apply_gradients(zip(grads, [gamma, beta]))
        beta = tf.Variable(tf.clip_by_value(beta, 0., float("inf")), dtype=tf.dtypes.float32)
        
    return gamma, beta

def visualize_rvs_quantile(dist_type, gamma, beta):
    """ 
    Visualizing quantile function of parametric distribution, results of "param_distribution_upr function".
    
    Estimated quantile function vs True quantile

    Args:
        dist_type ([str]): Generating distribution, Exponential: 'ex', Chi-squared: 'chi', GEV: 'gev'
        size ([int]): Sample size
        seed ([int]): Seed number
        epoch ([int]): Iteration epochs

    """
    knots = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    
    if dist_type == 'ex':
        #sample = tf.constant(expon.rvs(loc=0, scale=0.5, size=size, random_state=seed), dtype=tf.float32)
        true_quantiles = expon.ppf(knots, loc=0, scale=0.5)
        
    elif dist_type == 'chi':
        #sample =  tf.constant(chi2.rvs(2, size=size, random_state=seed), dtype=tf.float32)
        true_quantiles = chi2.ppf(knots, 2)
    
    elif dist_type == 'tn':
        #sample = tf.constant(truncnorm.rvs(a=-2.5, b=0.5, loc=2.5, scale=1, size=size, random_state=seed), dtype=tf.float32)
        true_quantiles = truncnorm.ppf(knots, a=-2.5, b=0.5, loc=2.5, scale=1)
    
    elif dist_type == 'gev1':
        true_quantiles = genextreme.ppf(knots, loc=2, c=-0.3)
    
    else:
        #sample = tf.constant(genextreme.rvs(loc=2, c=-0.5, size=size, random_state=seed), dtype=tf.float32)
        true_quantiles = genextreme.ppf(knots, loc=2, c=0.5)
    
    mpl.style.use('seaborn')
    delta = tf.constant(np.append(np.append([0.0, 0.05, 0.05], np.repeat(1, 8)/10), [0.05, 0.05, 0.0]), dtype=tf.dtypes.float32)
    quantile_ = np.linspace(0.01, 1, 100, dtype=np.float32)
    estimated_return = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.constant(quantile_)))
    # empirical_quantile = np.quantile(sample, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    
    plt.plot(np.array([0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], dtype=np.float32),
             true_quantiles, 'b-', lw=3, label='True Quantiles')
    # plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], empirical_quantile, 'k.', ms=10, label='Empirical quantiles')
    plt.plot(np.linspace(0.0, 1, 100, dtype=np.float32), estimated_return.numpy(), 'r--', lw=3, label='Estimated Quantiles')
    plt.xlabel("Quantiles", fontsize=16)
    plt.ylabel("X", fontsize=16)
    plt.legend(fontsize=16)
    # plt.title("Estimated Quantile")
    plt.show()
    
def mvn_simul(seed, epoch, var_a, var_b, var_c, corr_ab, corr_ac, corr_bc, p=3):
    cov_ab = np.sqrt(var_a) * np.sqrt(var_b) * corr_ab 
    cov_ac = np.sqrt(var_a) * np.sqrt(var_c) * corr_ac
    cov_bc = np.sqrt(var_b) * np.sqrt(var_c) * corr_bc

    mv_rv = multivariate_normal([0.0, 0.0, 0.0], [[var_a, cov_ab, cov_ac], [cov_ab, var_b, cov_bc], [cov_ac, cov_bc, var_c]])
    
    arr_X = mv_rv.rvs(200, seed)
    X = tf.constant(arr_X)
    X = tf.cast(X, dtype=tf.float32)

    mu_hat = tf.reduce_mean(X, axis=0)[:, tf.newaxis]
    mu_hat_mean = tf.math.reduce_sum(mu_hat)

    tf.random.set_seed(1)

    p = X.shape[1]

    pi = tf.Variable(tf.constant([1/3, 1/3, 1/3])[:, tf.newaxis], dtype=tf.dtypes.float32)

    beta = tf.Variable(tf.random.uniform([11], minval=0.0, maxval=0.3, dtype=tf.dtypes.float32)[tf.newaxis,:])

    delta = tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.dtypes.float32)

    gamma = tf.Variable(tf.constant(np.array([-2.0]), dtype=tf.dtypes.float32))

    initial_learning_rate1 = 0.01

    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate1,
    decay_steps=epoch,
    decay_rate=0.96,
    staircase=True)
    
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule1)

    NUM_ITER = tqdm(range(epoch))

    P = matrix(np.eye(p))

    G = matrix(np.vstack([np.eye(p), -1 * np.eye(p)]))
    h = matrix(np.ones(2*p))

    for j in NUM_ITER:
        with tf.GradientTape(persistent=True) as tape:
            quantile_dl = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.math.cumsum(delta)))
            
            portfolio = tf.linalg.matmul(X, pi)
        
            upr_  = tf.map_fn(lambda x: upr(x, gamma, beta, delta, quantile_dl), portfolio)
                        
            loss = tf.reduce_mean(upr_)

            if (j==0) :
                NUM_ITER.set_postfix({'loss': loss.numpy()})
            
            if (j+1) % 5 == 0:
                NUM_ITER.set_postfix({'loss': loss.numpy()})
                
        grads = tape.gradient(loss, [pi, gamma, beta])        
        optimizer1.apply_gradients(zip(grads, [pi, gamma, beta]))
        
        A = matrix(np.vstack([mu_hat.numpy().transpose(), np.ones(p).transpose()]))
        q = matrix((-1)*pi.numpy().astype(np.double))
        b = matrix(np.array([0.0, 1.0])[:, np.newaxis].astype(np.double))
        
        sol = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        pi_tmp = np.array(sol['x'])   
        pi = tf.Variable(pi_tmp, dtype=tf.float32)     
        beta = tf.Variable(tf.clip_by_value(beta, 0., float("inf")), dtype=tf.dtypes.float32)
        
    return pi, arr_X