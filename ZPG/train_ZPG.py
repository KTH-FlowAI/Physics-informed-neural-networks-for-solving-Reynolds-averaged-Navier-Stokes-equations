import numpy as np
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN_ZPG import PINNs
from mat73 import loadmat

from matplotlib import pyplot as plt
from time import time

def l2norm_err(ref, pred):
    return np.linalg.norm(ref - pred, axis = (1, 2)) / np.linalg.norm(ref, axis = (1, 2)) * 100
#%% DATA
les = loadmat('data/ZPG_EAdiag.mat')
les = les['ZPG_EA_diag']

Retheta = les['Retheta']
ind = np.isnan(Retheta)

Retheta = Retheta[~ind]
i_trn = (Retheta > 1000) & (Retheta < 7000) 
i_trn = np.squeeze(i_trn)
id_yn = 200

u = les['U'][:id_yn, ~ind][:, i_trn]
v = les['V'][:id_yn, ~ind][:, i_trn]
uv = les['uv'][:id_yn, ~ind][:, i_trn]

uu = les['uu'][:id_yn, ~ind][:, i_trn]
vv = les['vv'][:id_yn, ~ind][:, i_trn]

x = les['x'][:id_yn, ~ind][:, i_trn]
y = les['y'][:id_yn, ~ind][:, i_trn]
#%%
x = x - x.min()
y = y - y.min()
ny, nx = x.shape
#%% collocation points

ref = np.stack((u, v, uv, uu, vv))
#%% params
ac = 'tanh'
act = activations.tanh

nn = 20
nl = 8
n_adam = 1000
n_cp = 5000

#%%
lb = np.array([x.min(), y.min()])
ub = np.array([x.max(), y.max()])

# cp = lb + (ub-lb) * lhs(2, n_cp)
cp = np.concatenate((x[:, ::500].reshape((-1, 1)), y[:, ::500].reshape((-1, 1))), axis = 1)
n_cp = len(cp)

#%%
ind_bc = np.zeros(x.shape, dtype = bool)
ind_bc[[0, -1], ::20] = True
ind_bc[:, [0, -1]] = True

x_bc = x[ind_bc].flatten()
y_bc = y[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()
uv_bc = uv[ind_bc].flatten()
uu_bc = uu[ind_bc].flatten()
vv_bc = vv[ind_bc].flatten()

bc = np.array([x_bc, y_bc, u_bc, v_bc, uv_bc, uu_bc, vv_bc]).T

ni = 2
nv = bc.shape[1] - ni

p = 1
indx_bc = np.random.choice([False, True], len(bc), p=[1 - p, p])
bc = bc[indx_bc]

n_bc = len(bc)
test_name = f'_{nn}_{nl}_{ac}_{n_adam}_{n_cp}_{n_bc}'

#%%

inp = layers.Input(shape = (ni,))
hl = inp
for i in range(nl):
    hl = layers.Dense(nn, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
st_time = time()

pinn = PINNs(model, opt, n_adam)
hist = pinn.fit(bc, cp)

en_time = time()
comp_time = en_time - st_time
# %%
cpp = np.array([x.flatten(), y.flatten()]).T

pred = pinn.predict(cpp)
u_p = pred[:, 0].reshape(u.shape)
v_p = pred[:, 1].reshape(u.shape)
uv_p = pred[:, 2].reshape(u.shape)
uu_p = pred[:, 3].reshape(u.shape)
vv_p = pred[:, 4].reshape(u.shape)

pred = np.stack((u_p, v_p, uv_p, uu_p, vv_p))
#%%
plt.contourf(x, y, u)
plt.figure()
plt.contourf(x, y, u_p)
plt.figure()
plt.contourf(x, y, vv)
plt.figure()
plt.contourf(x, y, vv_p)
# %%
err = l2norm_err(ref, pred)
print(err)
#%%
n = int(u.shape[1]/2)
plt.figure()
plt.plot(y[:, n], u[:, n])
plt.plot(y[:, n], u_p[:, n])
plt.xscale('log')
plt.figure()
plt.plot(y[:, n], uv[:, n])
plt.plot(y[:, n], uv_p[:, n])
plt.xscale('log')
#%%
np.savez_compressed('res' + test_name, pred = pred, ref = ref, x = x, y = y, hist = hist, err = err, ct = comp_time)
model.save('model' + test_name + '.h5')

