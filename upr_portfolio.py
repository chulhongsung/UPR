import os
import numpy as np
import pandas as pd

import tensorflow as tf
from utils import *

DATA_PATH = os.getcwd() + "/data.npy"

X_total_ = np.load(DATA_PATH)

X_total = np.delete(X_total_, 122, 1)

N = X_total.shape[0]
p = X_total.shape[1]

df_pi = pd.DataFrame(columns=np.arange(p))
df_pi = df_pi.add_prefix('pi_')

gamma_name_lst = ['gamma']
beta_name_lst = ['beta' + str(i) for i in range(17)]
gamma_name_lst.extend(beta_name_lst)

name_lst = gamma_name_lst

df_param = pd.DataFrame(columns=name_lst)

i = 1

while (240+60*(i-1)) <= N:
    
    x  = tf.constant(X_total[(60*(i-1)):(240+60*(i-1)),:], dtype=tf.float32)

    mu_hat = tf.reduce_mean(x, axis=0)[:, tf.newaxis]
    mu_market = tf.reduce_mean(x)
    mu_hat_squad = tf.linalg.matmul(mu_hat, mu_hat, transpose_a=True)
    mu_hat_sum = tf.math.reduce_sum(mu_hat)
    tmu = mu_market + 0.005
    pi = tf.Variable(tf.random.uniform([p], minval=0, maxval=0.1, dtype=tf.dtypes.float32)[:, tf.newaxis])

    beta = tf.Variable(tf.random.uniform([17], minval=0.0, maxval=0.1, dtype=tf.dtypes.float32)[tf.newaxis,:])

    delta = tf.constant(np.append([0.0], np.repeat(1, 10)/10), dtype=tf.dtypes.float32)

    gamma = tf.Variable(tf.constant(np.array([-0.1]), dtype=tf.dtypes.float32))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for j in range(1000):
        with tf.GradientTape() as tape:
            quantile_dl = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.math.cumsum(delta)))
            
            portfolio = tf.linalg.matmul(x, pi)
        
            upr_  = tf.map_fn(lambda x: upr(x, gamma, beta, delta, quantile_dl), portfolio)
                        
            loss = tf.reduce_mean(upr_)

            if j == 0:
                print(loss)

            if (j+1) % 100 == 0:
                print(loss)
            
        grads = tape.gradient(loss, [pi, gamma, beta])
        optimizer.apply_gradients(zip(grads, [pi, gamma, beta]))

        beta = tf.Variable(tf.clip_by_value(beta, 0., float("inf")), dtype=tf.dtypes.float32)
        eta_1 = (mu_hat_squad*(tf.math.reduce_sum(pi) - 1) - mu_hat_sum * (tf.linalg.matmul(mu_hat, pi, transpose_a=True) - tmu))/(p*mu_hat_squad - mu_hat_sum**2)
        eta_2 = (mu_hat_sum * tf.math.reduce_sum(pi) - mu_hat_sum - p * tf.linalg.matmul(mu_hat, pi, transpose_a=True) + p * tmu)/(mu_hat_sum**2 - p*mu_hat_squad)
        pi =  tf.Variable(pi - eta_1 - eta_2 * mu_hat)

    tmp_pi = pd.DataFrame(np.squeeze(pi.numpy()))
    tmp_pi = tmp_pi.transpose().add_prefix('pi_')
    df_pi = pd.concat([df_pi,tmp_pi],axis=0) 
    
    tmp_param = pd.DataFrame(np.concatenate([gamma.numpy(), np.squeeze(beta.numpy())]))
    tmp_param = tmp_param.transpose()
    tmp_param.columns = name_lst
    df_param = pd.concat([df_param, tmp_param], axis=0)
    
    i = i + 1
