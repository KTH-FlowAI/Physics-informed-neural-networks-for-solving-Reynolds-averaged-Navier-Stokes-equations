import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from lbfgs import optimizer as lbfgs_op

class PINNs(models.Model):
    def __init__(self, model, optimizer, epochs, scaling_params, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.sopt = lbfgs_op(self.trainable_variables)
        self.nu = 5e-6
        self.scaling_params = tf.convert_to_tensor(scaling_params, tf.float32)
     
    @tf.function
    def cord_trans(self, xy):
        yn = xy[:, 1:2] * self.scaling_params[-1]
        xn = xy[:, 0:1] * self.scaling_params[-2]
        alpha = xy[:, 2:3]
        xa = xy[:, 3:4]
        ya = xy[:, 4:5]
        y = yn * tf.cos(alpha) + ya
        x = - yn * tf.sin(alpha) + xa
        return x, y
              
    @tf.function
    def net_f(self, cp):
        x, y = self.cord_trans(cp)
        alpha = cp[:, 2:3]
        xa = cp[:, 3:4]
        ya = cp[:, 4:5]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            
            yn = (y - ya) / tf.cos(alpha)
            xn = x + yn * tf.sin(alpha)
            X = tf.concat([xn, yn], axis=1) / self.scaling_params[-2:]

            UV = self.model(X)
            U = UV[:, :1] * self.scaling_params[0]
            V = UV[:, 1:2] * self.scaling_params[1]
            uv = UV[:, 2:3] * self.scaling_params[2]
            uu = UV[:, 3:4] * self.scaling_params[3]
            vv = UV[:, 4:5] * self.scaling_params[4]
            P = UV[:, 5:6] * self.scaling_params[5]
           
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
        
        f1 = U * U_x + V * U_y + P_x - self.nu * (U_xx + U_yy) + uu_x + uv_y
        f2 = U * V_x + V * V_y + P_y - self.nu * (V_xx + V_yy) + uv_x + vv_y
        f3 = U_x + V_y

        return tf.concat([f1, f2, f3], axis = 1)
    
    
    @tf.function
    def train_step(self, bc, cp):
        X = bc[:, :2]
        y = bc[:, 2:]
        with tf.GradientTape() as tape:
            u_p_bc = self.model(X)
            
            f = self.net_f(cp)
            
            loss_bc = tf.reduce_mean(tf.square(y[:, :-1] - u_p_bc[:, :-1]))
            loss_f = tf.reduce_mean(tf.square(f))
            
            loss_u = loss_bc
            loss = loss_u + loss_f
            
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)
        return loss, grads, tf.stack([l1, l2, l3])
    
    def fit(self, bc, cp):
        bc = tf.convert_to_tensor(bc, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
            
            
        self.sopt.minimize(func)
            
        return np.array(self.hist)
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        u_p = self.model(cp)
        return u_p.numpy()