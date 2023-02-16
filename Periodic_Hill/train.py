#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, initializers
from PINN import PINNs
from matplotlib import pyplot as plt
#%%
d = np.load('./../data/stsphill.npz')
pos = d['pos']
scal = d['scal']

x = pos[0]
ind = (x[0] > 1) & (x[0] < 5)

u = scal[0][:, ind]
v = scal[1][:, ind]
p = scal[3][:, ind]
uv = scal[8][:, ind] - u*v
uu = scal[4][:, ind] - u**2
vv = scal[5][:, ind] - v**2

x = pos[0][:, ind]
y = pos[1][:, ind]

u[0] = 0.0
v[0] = 0.0
uv[0] = 0.0
uu[0] = 0.0
vv[0] = 0.0
#%%
x = x - x.min()
y = y - y.min()
x_m = x.max()
y_m = y.max()
x = x / x_m
y = y / y_m


i_BC = np.zeros(x.shape, dtype = bool)
i_BC[0] = True; i_BC[-1] = True
i_BC[:, 0] = True; i_BC[:, -1] = True

# np.random.seed(24)
# i_f = np.random.choice(a = [False, True], size = x.shape, p = [0.99, 0.01])
#%%

xy_bc = np.concatenate((x[i_BC].reshape((-1, 1)),
                          y[i_BC].reshape((-1, 1))), axis = 1)

uv_bc = np.concatenate((u[i_BC].reshape((-1, 1)),
                        v[i_BC].reshape((-1, 1)),
                        uv[i_BC].reshape((-1, 1)),
                        uu[i_BC].reshape((-1, 1)),
                        vv[i_BC].reshape((-1, 1)),
                        p[i_BC].reshape((-1, 1))), axis = 1)

xy_f = np.concatenate((x[::10, ::2].reshape((-1, 1)),
                       y[::10, ::2].reshape((-1, 1))), axis = 1)

out_m = np.abs(uv_bc).max(axis = 0) 
uv_bc = uv_bc / out_m

u_m = out_m[0]
v_m = out_m[1]
uv_m = out_m[2]
uu_m = out_m[3]
vv_m = out_m[4]
p_m = out_m[5]
m = np.array([u_m, v_m, uv_m, uu_m, vv_m, p_m, x_m, y_m])
#%%
initializer = tf.keras.initializers.glorot_uniform()
act = tf.keras.activations.tanh
inp_U = layers.Input(shape = (2,), name = 'inp_U')
h_l = inp_U
for i in range(8):
    h_l = layers.Dense(20, activation = act, kernel_initializer = initializer)(h_l)
U_l = layers.Dense(6, kernel_initializer = initializer, name = 'U')(h_l)

U_net = models.Model(inputs = inp_U, outputs = U_l)
#%%
epochs = 100
opt = optimizers.Adam(lr = 1e-3)
pinns = PINNs(U_net, opt, epochs, m, sopt = 'BFGS', a = [1.0, 1.0])
pinns.fit(xy_bc, uv_bc, xy_f)

hist = np.array(pinns.hist)
np.save('hist_2', hist)
U_net.save('model_2.h5')
# %%
xy_p = np.concatenate((x.reshape((-1, 1)),
                        y.reshape((-1, 1))), axis = 1)

u_pred = U_net.predict(xy_p)
u_p = u_pred[:, 0].reshape(u.shape) * u_m
v_p = u_pred[:, 1].reshape(u.shape) * v_m
uv_p = u_pred[:, 2].reshape(u.shape) * uv_m
uu_p = u_pred[:, 3].reshape(u.shape) * uu_m
vv_p = u_pred[:, 4].reshape(u.shape) * vv_m
p_p = u_pred[:, 5].reshape(u.shape) * p_m
#%%
plt.contourf(x, y, u)
plt.figure()
plt.contourf(x, y, u_p)
plt.figure()
plt.contourf(x, y, vv)
plt.figure()
plt.contourf(x, y, vv_p)
# %%
e_u = np.linalg.norm(u - u_p)/np.linalg.norm(u); print(e_u)
e_v = np.linalg.norm(v - v_p)/np.linalg.norm(v); print(e_v)
e_p = np.linalg.norm(p - p_p)/np.linalg.norm(p); print(e_p)
e_uv = np.linalg.norm(uv - uv_p)/np.linalg.norm(uv); print(e_uv)
e_uu = np.linalg.norm(uu - uu_p)/np.linalg.norm(uu); print(e_uu)
e_vv = np.linalg.norm(vv - vv_p)/np.linalg.norm(vv); print(e_vv)
#%%
n = int(u.shape[1]/2)
plt.figure()
plt.plot(y[:, n], u[:, n])
plt.plot(y[:, n], u_p[:, n])
plt.figure()
plt.plot(y[:, n], uv[:, n])
plt.plot(y[:, n], uv_p[:, n])
#%%
np.savez_compressed('./test_2', u_p = u_p, v_p = v_p, p_p = p_p
                            , uu_p = uu_p, uv_p = uv_p, vv_p = vv_p)

# %%
