import numpy as np
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN import PINNs

from matplotlib import pyplot as plt
from time import time

def l2norm_err(ref, pred):
    return np.linalg.norm(ref - pred, axis = (1, 2)) / np.linalg.norm(ref, axis = (1, 2)) * 100
#%% DATA
d = np.load('./../finished_tests/data/top2n_HighRes.npz')

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
# yn = read('yn')

x = read('x').T
y = read('y').T
# alpha = read('alpha')

# y = yn.copy()
# x = np.repeat(xa.reshape((-1, 1)), x.shape[1], axis = 1)
# xa = np.repeat(xa.reshape((-1, 1)), x.shape[1], axis = 1)
# ya = np.repeat(ya.reshape((-1, 1)), x.shape[1], axis = 1)

x = x - x.min()
y = y - y.min()
#%% collocation points

ref = np.stack((u, v, p, uv, uu, vv))
#%% params
ac = 'tanh'
act = activations.tanh

nn = 20
nl = 8
n_adam = 1000
n_cp = 5000


lb = np.array([x.min(), y.min()])
ub = np.array([x.max(), y.max()])

# cp = lb + (ub-lb) * lhs(2, n_cp)
cp = np.concatenate((x[:, ::500].reshape((-1, 1)), y[:, ::500].reshape((-1, 1))), axis = 1)
n_cp = len(cp)
#%%
ind_bc = np.zeros(x.shape, dtype = bool)
ind_bc[[0, -1], ::50] = True
ind_bc[:, [0, -1]] = True

x_bc = x[ind_bc].flatten()
y_bc = y[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()
uv_bc = uv[ind_bc].flatten()
uu_bc = uu[ind_bc].flatten()
vv_bc = vv[ind_bc].flatten()
p_bc = p[ind_bc].flatten()

bc = np.array([x_bc, y_bc, u_bc, v_bc, uv_bc, uu_bc, vv_bc, p_bc]).T

ni = 2
nv = bc.shape[1] - ni

pr = 1
indx_bc = np.random.choice([False, True], len(bc), p=[1 - pr, pr])
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
p_p = pred[:, 5].reshape(u.shape)

pred = np.stack((u_p, v_p, p_p, uv_p, uu_p, vv_p))
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
plt.figure()
plt.plot(y[:, n], uv[:, n])
plt.plot(y[:, n], uv_p[:, n])
#%%
np.savez_compressed('res' + test_name, pred = pred, ref = ref, hist = hist, ct = comp_time, err = err)
model.save('model' + test_name + '.h5')

