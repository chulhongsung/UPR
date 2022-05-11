import tensorflow as tf

def linear_spline(alpha, gamma, beta, delta):
    mask = alpha - tf.math.cumsum(delta) >= 0
    
    bl = tf.concat([beta[:, 0][:, tf.newaxis], (beta[:, 1:] - beta[:,:-1])], axis=1)
    
    dl = tf.math.cumsum(delta)[tf.newaxis, :]
    
    z = gamma + tf.math.reduce_sum(bl *  tf.cast(mask, dtype=tf.float32)) * alpha - tf.math.reduce_sum((bl * tf.cast(mask, dtype=tf.float32)) * (dl * tf.cast(mask, dtype=tf.float32)))
    
    return z

def upr(z, gamma, beta, delta, quantile_dl):
    mask = z >= quantile_dl # l0를 찾기위한 mask
        
    bl = tf.squeeze(tf.concat([tf.constant(beta[:, 0], shape=(1,))[:, tf.newaxis], (beta[:,1:] - beta[:,:-1])], axis=1))

    dl = tf.math.cumsum(delta)

    mask_bl = tf.boolean_mask(bl, mask)

    mask_dl = tf.boolean_mask(dl, mask)
    
    tilde_a =  tf.cast(tf.clip_by_value((z - gamma + tf.math.reduce_sum(mask_bl * mask_dl)) / (tf.math.reduce_sum(mask_bl)+0.00000001), clip_value_min=0.0001, clip_value_max=1), dtype=tf.dtypes.float32)
    
    upr = (1+tf.math.log(tilde_a))*(z-gamma) + tf.math.reduce_sum(bl*(((-1/2) * ((1-dl)**2)) + 1 - tf.maximum(tilde_a, dl) + dl * tf.maximum(tf.math.log(tilde_a), tf.math.log(dl))))
    
    return upr
