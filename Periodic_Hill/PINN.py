import tensorflow as tf
from tensorflow.keras import models
import scipy.optimize as sopt
import numpy as np


class PINNs(models.Model):
    def __init__(self, net_U, optimizer, epochs, mu, sopt = None, a = [1, 1], **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.net_U = net_U
        self.optimizer = optimizer
        self.epochs = epochs
        self.a = a
        self.mu = tf.convert_to_tensor(mu, tf.float32)
        self.sopt = sopt
        self.itr = 0
        self.hist = []
        
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
    
    def cord_trans(self, xy):
        x = xy[:, :1] * self.mu[-2]
        y = xy[:, 1:2] * self.mu[-1]
        return x, y
       
    def net_f(self, xy):
        x, y = self.cord_trans(xy)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)

            X = tf.concat([x, y], axis=1) / self.mu[-2:]

            UV = self.net_U(X)
            U = UV[:, :1] * self.mu[0]
            V = UV[:, 1:2] * self.mu[1]
            uv = UV[:, 2:3] * self.mu[2]
            uu = UV[:, 3:4] * self.mu[3]
            vv = UV[:, 4:5] * self.mu[4]
            P = UV[:, 5:6] * self.mu[5]

           
            U_x = tape.gradient(U, x)
            U_y = tape.gradient(U, y)
            V_x = tape.gradient(V, x)
            V_y = tape.gradient(V, y)
        U_xx = tape.gradient(U_x, x)
        U_yy = tape.gradient(U_y, y)
        V_xx = tape.gradient(V_x, x)
        V_yy = tape.gradient(V_y, y)
        P_x = tape.gradient(P, x)
        P_y = tape.gradient(P, y)
        uv_y = tape.gradient(uv, y)
        uv_x = tape.gradient(uv, x)
        uu_x = tape.gradient(uu, x)
        vv_y = tape.gradient(vv, y)
        
      
              
        f1 = U * U_x + V * U_y + P_x - (1/2800) * (U_xx + U_yy) + uu_x + uv_y
        f2 = U * V_x + V * V_y + P_y - (1/2800) * (V_xx + V_yy) + uv_x + vv_y
        f3 = U_x + V_y

        return tf.concat([f1, f2, f3], axis = 1), tf.concat([U_x, U_y, V_x, V_y], axis = 1)

    def train_step(self, xy, uv, xy_f):
             
        with tf.GradientTape(persistent=True) as tape:
            f, _ = self.net_f(xy_f)
            uv_pred = self.net_U(xy)
            UV_loss = tf.reduce_mean(tf.square(uv - uv_pred))#, axis = 0)
            f_loss = tf.reduce_mean(tf.square(f))#, axis = 0)
            total_loss = self.a[0] * UV_loss + self.a[1] * f_loss
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        l1 = tf.reduce_mean(total_loss)
        l2 = tf.reduce_mean(UV_loss)
        l3 = tf.reduce_mean(f_loss)

        self.hist.append(np.array([l1, l2, l3]))
        
        self.itr += 1
        print("epoch: {}, L: {:.3e}, u_l: {:.3e}, f_l: {:.3e}"
              .format(self.itr, l1, l2, l3))
        
        return total_loss, gradients
    

        
    def fit(self, xy, uv, xy_f):
        xy = tf.convert_to_tensor(xy, tf.float32)
        uv = tf.convert_to_tensor(uv, tf.float32)
        xy_f = tf.convert_to_tensor(xy_f, tf.float32)
        
        def func(params_1d):
            self.assign_params(params_1d)
            loss, grads = self.train_step(xy, uv, xy_f)
            loss = tf.reduce_mean(loss)
            grads = tf.dynamic_stitch(self.idx, grads)
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            loss, grads = self.train_step(xy, uv, xy_f)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        if self.sopt != None:
            print('Minimize using the Scipy Optimizer')
            init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
            results = sopt.minimize(fun = func, 
                                x0 = init_params, 
                                method = self.sopt,
                                jac = True, options = {'iprint' : 0,
                                                       'maxiter': 900,
                                                       'maxfun' : 900,
                                                       'maxcor' : 50,
                                                       'maxls': 50,
                                                       'gtol': 1.0 * np.finfo(float).eps,
                                                       'ftol' : 1.0 * np.finfo(float).eps})

    def pred(self, xy):
        xy = tf.convert_to_tensor(xy, tf.float32)
        f, u = self.net_f(xy)
        return f.numpy(), u.numpy()
