import numpy as np
from tensorflow.keras import models, layers, optimizers, activations
from PINN_FS import PINNs
from matplotlib import pyplot as plt
from time import time
from train_configs import FS_config
from error import l2norm_err
#%% 
#################
# DATA loading
#################
d = np.load('data/FalknerSkan_n0.08.npz')

u = d['u'].T
v = d['v'].T
x = d['x'].T
y = d['y'].T
p = d['p'].T
x = x - x.min()
y = y - y.min()
ref = np.stack((u, v, p))
#%% 
#################
# Training Parameters
#################

act = FS_config.act
nn = FS_config.n_neural
nl = FS_config.n_layer
n_adam = FS_config.n_adam
#%%
#################
# Training Data
#################

# Collection points
np.random.seed(24)
indx_f = np.random.choice(a = [False, True], size = x.shape, p = [0.99, 0.01])
cp = np.concatenate((x[indx_f].reshape((-1, 1)), 
                     y[indx_f].reshape((-1, 1))), axis = 1)
n_cp = len(cp)

# Boundary points
ind_bc = np.zeros(x.shape, dtype = bool)
ind_bc[[0, -1], :] = True
ind_bc[:, [0, -1]] = True

x_bc = x[ind_bc].flatten()
y_bc = y[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()

bc = np.array([x_bc, y_bc, u_bc, v_bc]).T

ni = 2
nv = bc.shape[1] - ni + 1
pp = 1

# Randomly select half of Boundary points
indx_bc = np.random.choice([False, True], len(bc), p=[1 - pp, pp])
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
print(f"INFO: Start training case : {test_name}")
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
p_p = pred[:, 2].reshape(u.shape)

# Shift the pressure to the reference level before calculating the error
# Becacuse we only have pressure gradients in N-S eqs but not pressure itself in BC
deltap =p[0,0] - p_p[0,0]
p_p = p_p + deltap
pred = np.stack((u_p, v_p, p_p))
err = l2norm_err(ref, pred)
#%%
#################
# Save prediction and Model
#################
np.savez_compressed('pred/res_FS' + test_name, pred = pred, ref = ref, x = x, y = y, hist = hist, err = err, ct = comp_time)
model.save('models/model_FS' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")