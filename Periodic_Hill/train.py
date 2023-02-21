import numpy as np
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN_phill import PINNs
from train_configs import phill_config
from matplotlib import pyplot as plt
from time import time
from error import l2norm_err
"""
Train PINN for Periodic Hill

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
d = np.load('data/stsphill.npz')
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

ref = np.stack((u, v, p, uv, uu, vv))
#%% 
#################
# Training Parameters
#################
act = phill_config.act
nn = phill_config.n_neural
nl = phill_config.n_layer
n_adam = phill_config.n_adam

n_cp = 2430
tcp = 'grid'

cp = np.concatenate((x[::10, ::2].reshape((-1, 1)),
                       y[::10, ::2].reshape((-1, 1))), axis = 1)
#% Boundary points
ind_bc = np.zeros(x.shape, dtype = bool)
ind_bc[[0, -1]] = True
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
p = 1
# Randomly select half of Boundary points
indx_bc = np.random.choice([False, True], len(bc), p=[1 - p, p])
bc = bc[indx_bc]

test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{tcp}'
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
err = l2norm_err(ref, pred)
#%%
np.savez_compressed('pred/res_phill' + test_name, pred = pred, ref = ref, hist = hist, ct = comp_time)
model.save('models/model_phill' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")

