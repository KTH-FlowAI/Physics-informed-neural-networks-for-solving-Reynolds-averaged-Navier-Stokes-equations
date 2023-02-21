#%%
import numpy as np
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN_ZPG import PINNs
from scipy.io import loadmat
from matplotlib import pyplot as plt
from time import time
from error import l2norm_err
from train_configs import ZPG_config
"""
Train PINN for ZPG

Return: 
    1. A npz file contain: 
                          + Prediction "pred" : [u, v, uv]
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
les = loadmat('data/ZPG.mat')

Retheta = les['Retheta']
ind = np.isnan(Retheta)

Retheta = Retheta[~ind]
i_trn = (Retheta > 1000) & (Retheta < 7000) 
i_trn = np.squeeze(i_trn)
id_yn = 200

ind = ind.reshape(-1)
u = les['U'][:id_yn, ~ind][:, i_trn]
v = les['V'][:id_yn, ~ind][:, i_trn]
uv = les['uv'][:id_yn, ~ind][:, i_trn]

uu = les['uu'][:id_yn, ~ind][:, i_trn]
vv = les['vv'][:id_yn, ~ind][:, i_trn]

x = les['x'][:id_yn, ~ind][:, i_trn]
y = les['y'][:id_yn, ~ind][:, i_trn]
x = x - x.min()
y = y - y.min()
ny, nx = x.shape
ref = np.stack((u, v, uv))
#%%
#################
# Training Parameters
#################
act = ZPG_config.act
nn = ZPG_config.n_neural
nl = ZPG_config.n_layer
n_adam = ZPG_config.n_adam
cp_step = ZPG_config.cp_step

cp = np.concatenate((x[:, ::cp_step].reshape((-1, 1)), y[:, ::cp_step].reshape((-1, 1))), axis = 1)
n_cp = len(cp)

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

bc = np.array([x_bc, y_bc,
                u_bc, v_bc, 
                uv_bc]).T

ni = 2
nv = bc.shape[1] - ni

p = 1
indx_bc = np.random.choice([False, True], len(bc), p=[1 - p, p])
bc = bc[indx_bc]

test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}'
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

pred = np.stack((u_p, v_p, uv_p))
err = l2norm_err(ref, pred)
#%%
np.savez_compressed('pred/res_ZPG' + test_name, pred = pred, ref = ref, x = x, y = y, hist = hist, err = err, ct = comp_time)
model.save('models/model_ZPG' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")