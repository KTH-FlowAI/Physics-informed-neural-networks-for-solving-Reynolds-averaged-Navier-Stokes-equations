#%%
import numpy as np
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN_NACA import PINNs
from matplotlib import pyplot as plt
from time import time
from error import l2norm_err
from train_configs import NACA_config
"""
Train PINN for NACA4412 BL

Return: 
    1. A npz file contain: 
                          + Prediction "pred" : [u, v, uv, uu, vv, p]
                          + Reference   "ref"
                          + x grid      "x"
                          + y grid      "y"
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

u = read('U').T
v = read('V').T
p = read('P').T
uv = read('uv').T
uu = read('uu').T
vv = read('vv').T
x = read('x').T
y = read('y').T
x = x - x.min()
y = y - y.min()
#%% collocation points
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


cp = np.concatenate((x[:, ::cp_step].reshape((-1, 1)), y[:, ::cp_step].reshape((-1, 1))), axis = 1)
n_cp = len(cp)
#Boundary points 
ind_bc = np.zeros(x.shape, dtype = bool)
ind_bc[[0, -1], ::bc_step] = True
ind_bc[:, [0, -1]] = True

x_bc = x[ind_bc].flatten()
y_bc = y[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()
uv_bc = uv[ind_bc].flatten()
uu_bc = uu[ind_bc].flatten()
vv_bc = vv[ind_bc].flatten()
p_bc = p[ind_bc].flatten()

bc = np.array([x_bc, y_bc, 
               u_bc, v_bc, 
               uv_bc, uu_bc, vv_bc,
                 p_bc]).T

ni = 2
nv = bc.shape[1] - ni
pr = 1
# Randomly select half of Boundary points
indx_bc = np.random.choice([False, True], len(bc), p=[1 - pr, pr])
bc = bc[indx_bc]

n_bc = len(bc)
test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{n_bc}'
#%%
#################
# Compiling Model
#################
inp = layers.Input(shape = (ni,))
hl = inp
for i in range(nl):
    hl = layers.Dense(nn, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
pinn = PINNs(model, opt, n_adam)

#################
# Training Process
#################
print(f"INFO: Start training case_{test_name}")
st_time = time()
hist = pinn.fit(bc, cp)
en_time = time()
comp_time = en_time - st_time
# %%
#################
# Prediction
#################
cpp = np.array([x.flatten(), y.flatten()]).T

pred = pinn.predict(cpp)
u_p = pred[:, 0].reshape(u.shape)
v_p = pred[:, 1].reshape(u.shape)
uv_p = pred[:, 2].reshape(u.shape)
uu_p = pred[:, 3].reshape(u.shape)
vv_p = pred[:, 4].reshape(u.shape)
p_p = pred[:, 5].reshape(u.shape)

pred = np.stack((u_p, v_p, p_p, uv_p, uu_p, vv_p))
err = l2norm_err(ref,pred)
#%%
#################
# Save prediction and Model
#################
np.savez_compressed('pred/res_NACA' + test_name, pred = pred, ref = ref, hist = hist, ct = comp_time, err = err)
model.save('models/model_NACA' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")