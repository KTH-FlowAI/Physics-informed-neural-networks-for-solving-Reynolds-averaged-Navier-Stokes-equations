#%%
import numpy as np
from pyDOE import lhs
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations
from PINN_NACA import PINNs
from matplotlib import pyplot as plt
from time import time
from error import l2norm_err
from train_configs import NACA_config

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')
"""
Train PINN for NACA4412 BL

Return: 
    1. A npz file contain: 
                          + Prediction  "pred" : [u, v, uv, uu, vv, p]
                          + Reference   "ref"
                          + xn grid     "xn"
                          + yn grid     "yn"
                          + error       "error"
    
    2. A saved .h5 model of PINN 
    
"""
#%%
#################
# DATA loading
#################
d = np.load('data/top2n_HighRes.npz')

topy = 170
x = d['x'][:, :topy]
ind = (x[:, 0] > 0.5) & (x[:, 0] < 0.99)

def read(x):
    if len(d[x].shape) == 2:
        return d[x][ind, :topy]
    if len(d[x].shape) == 1:
        return d[x][ind]

u = read('U')
v = read('V')
p = read('P')
uv = read('uv')
uu = read('uu')
vv = read('vv')
yn = read('yn')

xa = read('xa')
ya = read('ya')
alpha = read('alpha')

xa = xa - xa.min()
ya = ya - ya.min()

xn = np.repeat(xa.reshape((-1, 1)), x.shape[1], axis = 1)
a = np.repeat(alpha.reshape((-1, 1)), x.shape[1], axis = 1)
xa = np.repeat(xa.reshape((-1, 1)), x.shape[1], axis = 1)
ya = np.repeat(ya.reshape((-1, 1)), x.shape[1], axis = 1)

ref = np.stack((u, v, p, uv, uu, vv))
#%%
#################
# Training Parameters
#################
act = NACA_config.act
nn = NACA_config.n_neural
nl = NACA_config.n_layer
n_adam = NACA_config.n_adam
cp_step = NACA_config.cp_step
bc_step = NACA_config.bc_step

xn = xn - xn.min()
yn = yn - yn.min()
x_m = xn.max()
y_m = yn.max()
xn = xn / x_m
yn = yn / y_m

#%% data on the domain boundaries
ind_bc = np.zeros(xn.shape, dtype = bool)
ind_bc[[0, -1], :] = True
ind_bc[::bc_step, [0, -1]] = True

xy_bc = np.concatenate((xn[ind_bc].reshape((-1, 1)),
                          yn[ind_bc].reshape((-1, 1))), axis = 1)

uv_bc = np.concatenate((u[ind_bc].reshape((-1, 1)),
                          v[ind_bc].reshape((-1, 1)),
                          uv[ind_bc].reshape((-1, 1)),
                          uu[ind_bc].reshape((-1, 1)),
                          vv[ind_bc].reshape((-1, 1)),
                          p[ind_bc].reshape((-1, 1))), axis = 1)

scaling_params = np.abs(uv_bc).max(axis = 0) 
uv_bc = uv_bc / scaling_params
u_m = scaling_params[0]
v_m = scaling_params[1]
uv_m = scaling_params[2]
uu_m = scaling_params[3]
vv_m = scaling_params[4]
p_m = scaling_params[5]
scaling_params = np.array([u_m, v_m, uv_m, uu_m, vv_m, p_m, x_m, y_m])

pr = 1
indx_bc = np.random.choice([False, True], len(xy_bc), p=[1 - pr, pr])
uv_bc = uv_bc[indx_bc]
xy_bc = xy_bc[indx_bc]
bc = np.concatenate((xy_bc, uv_bc), axis=1)
n_bc = len(bc)
#%% collocation points
cp = np.concatenate((xn[::500].reshape((-1, 1)),
                       yn[::500].reshape((-1, 1)),
                       a[::500].reshape((-1, 1)),
                       xa[::500].reshape((-1, 1)),
                       ya[::500].reshape((-1, 1))), axis = 1)
n_cp = len(cp)
#%%
#################
# Compiling Model
#################
ni = xy_bc.shape[1]
nv = uv_bc.shape[1]
test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{n_bc}'

inp = layers.Input(shape = (ni,))
hl = inp
for i in range(nl):
    hl = layers.Dense(nn, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
pinn = PINNs(model, opt, n_adam, scaling_params)
#%%
#################
# Training Process
#################
print(f"INFO: Start training case_{test_name}")
st_time = time()
pinn.fit(bc, cp)
hist = np.array(pinn.hist)
en_time = time()
comp_time = en_time - st_time
# %%
#################
# Prediction
#################
cpp = np.concatenate((xn.reshape((-1, 1)),
                        yn.reshape((-1, 1))), axis = 1)

pred = pinn.predict(cpp)
u_p = pred[:, 0].reshape(u.shape) * u_m
v_p = pred[:, 1].reshape(u.shape) * v_m
uv_p = pred[:, 2].reshape(u.shape) * uv_m
uu_p = pred[:, 3].reshape(u.shape) * uu_m
vv_p = pred[:, 4].reshape(u.shape) * vv_m
p_p = pred[:, 5].reshape(u.shape) * p_m

p_p = p_p - p_p.mean() + p.mean()
pred = np.stack((u_p, v_p, p_p, uv_p, uu_p, vv_p))
err = l2norm_err(ref, pred)
#%%
#################
# Save prediction and Model
#################
np.savez_compressed('pred/res_NACA' + test_name, pred = pred, ref = ref, hist = hist, ct = comp_time, err = err, xn = xn, yn = yn)
model.save('models/model_NACA' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")