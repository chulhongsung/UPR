import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def linear_spline(alpha, gamma, beta, delta):
    mask = alpha - tf.math.cumsum(delta) >= 0
    
    bl = tf.concat([beta[:, 0][:, tf.newaxis], (beta[:, 1:] - beta[:,:-1])], axis=1)
    
    dl = tf.math.cumsum(delta)[tf.newaxis, :]
    
    z = gamma + tf.math.reduce_sum(bl *  tf.cast(mask, dtype=tf.float32)) * alpha - tf.math.reduce_sum((bl * tf.cast(mask, dtype=tf.float32)) * (dl * tf.cast(mask, dtype=tf.float32)))
    
    return z

def upr(z, gamma, beta, delta, quantile_dl):
    mask = z >= quantile_dl
        
    bl = tf.squeeze(tf.concat([tf.constant(beta[:, 0], shape=(1,))[:, tf.newaxis], (beta[:,1:] - beta[:,:-1])], axis=1))

    dl = tf.math.cumsum(delta)

    mask_bl = tf.boolean_mask(bl, mask)

    mask_dl = tf.boolean_mask(dl, mask)
    
    tilde_a =  tf.cast(tf.clip_by_value((z - gamma + tf.math.reduce_sum(mask_bl * mask_dl)) / (tf.math.reduce_sum(mask_bl)+0.00000001), clip_value_min=0.0001, clip_value_max=1), dtype=tf.dtypes.float32)
    
    upr = (1+tf.math.log(tilde_a))*(z-gamma) + tf.math.reduce_sum(bl*(((-1/2) * ((1-dl)**2)) + 1 - tf.maximum(tilde_a, dl) + dl * tf.maximum(tf.math.log(tilde_a), tf.math.log(dl))))
    
    return upr

def empirical_upr(sort_return, num_obs=60, num_knots=10):
    """
        Calculate empirical UPR 

    
    Args:
        sort_return ([type]): [description]
        num_knots ([type]): [description]
    """

    w = np.log(np.arange(1,num_obs+1) / num_obs)
    
    w_star = w / np.sum(w)
    
    sort_return_ = sort_return.reshape((-1, int(len(sort_return)/num_knots)))
    return_sum_quantile = np.cumsum(np.sum(sort_return_, axis=-1))
    
    alpha_risk = return_sum_quantile/(len(sort_return)/num_knots * np.arange(1, 11))
    
    upr = np.sum(w_star * sort_return)
    
    return alpha_risk, upr

def quantile_matching_figure(data, param, pi, start, end):
    """
    Args:
    
    Returns:
        plot
    """
    # define subplot grid
    _, axs = plt.subplots(nrows=4, ncols=4, figsize=(28, 28))
    plt.subplots_adjust(hspace=0.2)
    # loop through tickers and axes
    for num_window, ax in zip(range(start, end+1), axs.ravel()):
        gamma = np.array([param.iloc[num_window-1,0]], np.float32)
        beta = np.array([param.iloc[num_window-1,1:]], np.float32)
        
        X = data[(60*(num_window-1)):(240+60*(num_window-1)),:]     
        out_X = data[(240+60*(num_window-1)):(240+60*num_window),:] 

        portfolio = X @ np.array([pi.iloc[num_window-1]], np.float32).transpose()
        out_portfolio = out_X @ np.array([pi.iloc[num_window-1]], np.float32).transpose()
        
        empirical_quantile = np.quantile(portfolio, np.linspace(0.0, 1, 11, dtype=np.float32))
        out_empirical_quantile = np.quantile(out_portfolio, np.linspace(0.0, 1, 11, dtype=np.float32))

        quantile_ = np.linspace(0.0, 1, 100, dtype=np.float32)
        estimated_return = tf.squeeze(tf.map_fn(lambda x: linear_spline(x, gamma, beta, delta), tf.constant(quantile_)))
        # filter df for ticker and plot on specified axes
        if (num_window % 4) == 1:
            ax.plot(np.linspace(0.0, 1, 100, dtype=np.float32), estimated_return.numpy(), 'g-', lw=3, label='Estimated Quantiles')
            ax.plot(np.linspace(0.0, 1, 11, dtype=np.float32), empirical_quantile, 'b--', lw=3, label='Empirical Quantiles')
            ax.plot(np.linspace(0.0, 1, 11, dtype=np.float32), out_empirical_quantile, 'm:', lw=3, label='Out-of-sample Quantiles')
        else: 
            ax.plot(np.linspace(0.0, 1, 100, dtype=np.float32), estimated_return.numpy(), 'g-', lw=3)
            ax.plot(np.linspace(0.0, 1, 11, dtype=np.float32), empirical_quantile, 'b--', lw=3)
            ax.plot(np.linspace(0.0, 1, 11, dtype=np.float32), out_empirical_quantile, 'm:', lw=3)
        
        ax.set_xlabel("Quantiles", fontsize=20)
        if (num_window % 4) == 1:
            ax.set_ylabel("Portfolio Return", fontsize=20)
        if (num_window == 1 )| (num_window  == 17):
            ax.legend(fontsize=20)

        # chart formattin
    plt.show()
