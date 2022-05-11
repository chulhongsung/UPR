import os
import numpy as np
import pandas as pd

import tensorflow as tf

from cvxopt import matrix, solvers

from utils import *

solvers.options['show_progress']=False

DATA_PATH = os.getcwd() + "/data.npy"
PARAM_PATH = os.getcwd() + "/param.csv"
PI_PATH = os.getcwd() + "/pi.csv"

X_total_ = np.load(DATA_PATH)
param_init = pd.read_csv(PARAM_PATH)
pi_init = pd.read_csv(PI_PATH)

X_total = np.delete(X_total_, 122, axis=1)
N = X_total.shape[0]
p = X_total.shape[1]

lambda_1 = 0.01

df_pi = pd.DataFrame(columns=np.arange(162))
df_pi = df_pi.add_prefix('pi_')

gamma_name_lst = ['gamma']
beta_name_lst = ['beta' + str(i) for i in range(17)]
gamma_name_lst.extend(beta_name_lst)

name_lst = gamma_name_lst

df_param = pd.DataFrame(columns=name_lst)

D = np.hstack([np.eye(p), -1*np.eye(p)])

P = matrix(D.transpose() @ D)

G = matrix(-1*np.eye(2*p))

h = matrix(np.zeros(2*p))

i = 1

while (240+60*(i-1)) <= N:
    
    x  = tf.constant(X_total[(60*(i-1)):(240+60*(i-1)),:], dtype=tf.float32)
    mu_market = tf.reduce_mean(x)
    mu_hat = tf.reduce_mean(x, axis=0)[:, tf.newaxis]
    mu_hat_squad = tf.linalg.matmul(mu_hat, mu_hat, transpose_a=True)
    mu_hat_sum = tf.math.reduce_sum(mu_hat)
    tmu = mu_market + 0.005
   
    A = matrix(np.vstack((mu_hat.numpy().transpose() @ np.hstack([np.eye(p), -1*np.eye(p)]), np.hstack([np.ones((1, p), np.float32), -1*np.ones((1, p), np.float32)]), np.ones((1, 2*p), np.float32))))

    pi = tf.Variable(np.array([pi_init.iloc[i-1]], np.float32).transpose())
    
    beta = tf.Variable(np.array(param_init.iloc[i-1, 1:], dtype=np.float32)[tf.newaxis,:])

    delta = tf.constant(np.append(np.append([0.0, 0.025, 0.025, 0.025, 0.025], np.repeat(1, 8)/10), [0.025, 0.025, 0.025, 0.025]), dtype=tf.dtypes.float32)
    
    gamma = tf.Variable(tf.constant(np.array(param_init.iloc[i-1, 0]), dtype=tf.dtypes.float32))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for j in range(70):
        with tf.GradientTape() as tape:
            quantile_dl = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.math.cumsum(delta)))
            
            portfolio = tf.linalg.matmul(x, pi)
        
            upr_  = tf.map_fn(lambda x: upr(x, gamma, beta, delta, quantile_dl), portfolio)
                        
            loss = tf.reduce_mean(upr_) + lambda_1 * tf.norm(pi, ord=1)

            if j == 0:
                print(loss)
                 
            if (j+1) % 10 == 0:
                print(loss)

        grads = tape.gradient(loss, [pi, gamma, beta])
        optimizer.apply_gradients(zip(grads, [pi, gamma, beta]))

        q = matrix((-1)*(pi.numpy().transpose() @ D).transpose())
        b = matrix(np.array([tmu.numpy().item(), 1, np.round(tf.reduce_sum(tf.abs(pi)).numpy().item(), 4)]))
        
        sol = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        pi_tmp = np.array(sol['x'])
        pi = tf.Variable(np.hstack([np.eye(p), -1*np.eye(p)]) @ pi_tmp, dtype=tf.float32)
        
        beta = tf.Variable(tf.clip_by_value(beta, 0., float("inf")), dtype=tf.dtypes.float32)
    
    tmp_pi = pd.DataFrame(np.squeeze(pi.numpy()))
    tmp_pi = tmp_pi.transpose().add_prefix('pi_')
    df_pi = pd.concat([df_pi,tmp_pi],axis=0) 
    
    tmp_param = pd.DataFrame(np.concatenate([np.array(gamma.numpy())[np.newaxis,...], np.squeeze(beta.numpy())]))
    tmp_param = tmp_param.transpose()
    tmp_param.columns = name_lst
    df_param = pd.concat([df_param, tmp_param], axis=0)
    
    i = i + 1