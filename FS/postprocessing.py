#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from error import l2norm_err
from train_configs import FS_config
from matplotlib.ticker import ScalarFormatter
"""
Postprocessing of prediction data 

Return: 
    1. Loss vs Epoch
    2. U contour: Prediction, Reference, L1-norm
    3. V contour: Prediction, Reference, L1-norm
    4. P contour: Prediction, Reference, L1-norm
    5. Print l2-norm error 
"""
def err(y, yp):
    return np.abs(y - yp)

cmp = sns.color_palette('RdBu_r', as_cmap=True)
plt.set_cmap(cmp)
#%%
act = FS_config.act
nn = FS_config.n_neural
nl = FS_config.n_layer
n_adam = FS_config.n_adam
data_name = f"FS_{nn}_{nl}_{act}_{n_adam}_1035_1400"
data = np.load(f"pred/res_{data_name}.npz")
d = np.load('data/FalknerSkan_n0.08.npz')
x = d['x'].T
y = d['y'].T
# %%
plt.semilogy(data["hist"][:,0],label="Total")
plt.semilogy(data["hist"][:,1],label="Bc")
plt.semilogy(data["hist"][:,2],label="residual")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"figs/{data_name}_Loss")
# %%
pred = data["pred"]
ref = data["ref"]
names = ["U","V","P"]

u_p = pred[0]
u = ref[0]

v_p = pred[1]
v = ref[1]

p_t = pred[2]
p_p = ref[2]

eu = err(u, u_p)
ev = err(v, v_p)
ep = err(p_t, p_p)
# %%
fig, ax = plt.subplots(3, 3, sharex = True, sharey = True, figsize = (10, 5))
plt.set_cmap('Spectral')

c0 = ax[0, 0].contourf(x, y, u_p, levels = 16)
c1 = ax[1, 0].contourf(x, y, v_p, levels = 16)
c2 = ax[2, 0].contourf(x, y, p_p, levels = 16)


cbar1 = fig.colorbar(c0, ax = ax[0, 1], orientation = 'vertical', pad = 0.1, format = '%.2f')
cbar1.ax.locator_params(nbins = 3)
cbar2 = fig.colorbar(c1, ax = ax[1, 1], orientation = 'vertical', pad = 0.1, format = '%.2f')
cbar2.ax.locator_params(nbins = 3)


ax[0, 0].set_title('$U$ prediction')
ax[0, 1].set_title('$U$ reference')

cs0 = ax[0, 1].contourf(x, y, u, levels = 16)
cs1 = ax[1, 1].contourf(x, y, v, levels = 16)
cs2 = ax[2, 1].contourf(x, y, p_t, levels = 16)

cbar3 = fig.colorbar(c2, ax = ax[2, 1], orientation = 'vertical', pad = 0.1, format = '%.2f')
cbar3.ax.locator_params(nbins = 3)

c02 = ax[0, 2].contourf(x, y, eu, levels = 16)
c12 = ax[1, 2].contourf(x, y, ev, levels = 16)
c22 = ax[2, 2].contourf(x, y, ep, levels = 16)

fmt = ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))

cbar02 = fig.colorbar(c02, ax = ax[0, 2], orientation = 'vertical', pad = 0.1, format = fmt)
cbar02.ax.locator_params(nbins = 3)

cbar12 = fig.colorbar(c12, ax = ax[1, 2], orientation = 'vertical', pad = 0.1, format = fmt)
cbar12.ax.locator_params(nbins = 3)

cbar22 = fig.colorbar(c22, ax = ax[2, 2], orientation = 'vertical', pad = 0.1, format = fmt)
cbar22.ax.locator_params(nbins = 3)


cbs = fig.colorbar(cs0, ax = ax[0, 0], orientation = 'vertical', pad = 0.1, format = '%.2f')
cbs.ax.locator_params(nbins = 3)

cbs = fig.colorbar(cs1, ax = ax[1, 0], orientation = 'vertical', pad = 0.1, format = '%.2f')
cbs.ax.locator_params(nbins = 3)

cbs = fig.colorbar(cs2, ax = ax[2, 0], orientation = 'vertical', pad = 0.1, format = '%.2f')
cbs.ax.locator_params(nbins = 3)

ax[0, 2].set_title('$|$ref. - pred.$|$')


ax[1, 0].set_title('$V$ prediction')
ax[1, 1].set_title('$V$ reference')

ax[2, 0].set_title('$P$ prediction')
ax[2, 1].set_title('$P$ reference')

ax[0, 0].set_ylabel('$y$')
ax[1, 0].set_ylabel('$y$')
ax[2, 0].set_ylabel('$y$')

ax[2, 0].set_xlabel('$x$')
ax[2, 1].set_xlabel('$x$')


fig.tight_layout()

plt.savefig('./figs/FS.png', dpi = 300, bbox_inches='tight')
# %%
err = l2norm_err(ref, pred)
print(f"Error U = {np.round(err[0],3)}%\nError V = {np.round(err[1],3)}%\nError P = {np.round(err[2],3)}%")
# %%
