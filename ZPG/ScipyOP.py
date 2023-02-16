import tensorflow as tf
from tensorflow.keras import models
import scipy.optimize as sopt
import numpy as np


class optimizer(models.Model):
    def __init__(self, model, method = 'L-BFGS-B', **kwargs):
        super(optimizer, self).__init__(**kwargs)
        self.model = model
        self.method = method
        
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)

        count = 0
        idx = [] # stitch indices
        part = [] # partition indices
    
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n
    
        self.part = tf.constant(part)
        self.idx = idx
    
    def assign_params(self, params_1d):
        params_1d = tf.cast(params_1d, dtype = tf.float32)
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))       
    
    def minimize(self, func):
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
        results = sopt.minimize(fun = func, 
                            x0 = init_params, 
                            method = self.method,
                            jac = True, options = {'iprint' : 0,
                                                   'maxiter': 50000,
                                                   'maxfun' : 50000,
                                                   'maxcor' : 50,
                                                   'maxls': 50,
                                                   'gtol': 1.0 * np.finfo(float).eps,
                                                   'ftol' : 1.0 * np.finfo(float).eps})


# def func(params_1d):
#     self.assign_params(params_1d)
#     loss, grads = self.train_step(xy, uv, xy_f)
#     grads = tf.dynamic_stitch(self.idx, grads)
#     return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)