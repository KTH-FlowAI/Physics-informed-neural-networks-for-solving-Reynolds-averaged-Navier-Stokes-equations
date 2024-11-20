#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from d99 import rep_d99, rep_grad
from error import l2norm_err
from train_configs import NACA_config
"""
Postprocessing of prediction data 

Return: 
    1. Inner-scaled streamwise velocity and Reynolds-stress profiles
    2. Loss vs Epoch
    3. Print l2-norm error 
"""

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)

act = NACA_config.act
nn = NACA_config.n_neural
nl = NACA_config.n_layer
n_adam = NACA_config.n_adam

data_name = f"NACA_{nn}_{nl}_{act}_{n_adam}_2550_620"
d = np.load(f"pred/res_{data_name}.npz")

nu = 5e-6
u_p = d['pred'][0]
u = d['ref'][0]
y = d['yn']
uv_p = d['pred'][3]
uv = d['ref'][3]
uu_p = d['pred'][4]
uu = d['ref'][4]
vv_p = d['pred'][5]
vv = d['ref'][5]

k = 4
n = int(u.shape[0]/k)

p = rep_d99(y, u_p)
t = rep_d99(y, u)

H12_p = p[:, -1]
H12_t = t[:, -1]

Ue_p = p[:, 2]
Ue_t = t[:, 2]

theta_p = p[:, -2]
theta_t = t[:, -2]

Retheta_p = Ue_p * theta_p / nu
Retheta_t = Ue_t * theta_t / nu

uy = rep_grad(u, y)
uy_p = rep_grad(u_p, y)

utau_t = np.sqrt(nu * np.abs(uy[:, 0]))
utau_p = np.sqrt(nu * np.abs(uy_p[:, 0]))

lstar_p = nu / utau_p
lstar_t = nu / utau_t

cf_t = utau_t**2 / (0.5 * Ue_p**2)
cf_p = utau_p**2 / (0.5 * Ue_p**2)
#%%
fig, ax = plt.subplots(2, 2, sharex = True, figsize = (8, 6))
N = k - 1
colors = plt.cm.RdPu(np.linspace(0.4, 0.9, N))
for i in range(1, k):
    m = i * n
    ax[0, 0].plot(y[m]/lstar_t[m], u[m]/utau_t[m], c = colors[i - 1])
    ax[0, 1].plot(y[m]/lstar_t[m], uv[m]/utau_t[m]**2, c = colors[i - 1])
    ax[1, 0].plot(y[m]/lstar_t[m], uu[m]/utau_t[m]**2, c = colors[i - 1])
    ax[1, 1].plot(y[m]/lstar_t[m], vv[m]/utau_t[m]**2, c = colors[i - 1])

    ax[0, 0].plot(y[m]/lstar_p[m], u_p[m]/utau_p[m], c = colors[i - 1], ls = '--')
    ax[0, 1].plot(y[m]/lstar_p[m], uv_p[m]/utau_p[m]**2, c = colors[i - 1], ls = '--')
    ax[1, 0].plot(y[m]/lstar_t[m], uu_p[m]/utau_p[m]**2, c = colors[i - 1], ls = '--')
    ax[1, 1].plot(y[m]/lstar_t[m], vv_p[m]/utau_p[m]**2, c = colors[i - 1], ls = '--')

ax[0, 0].grid(which = 'major')
ax[0, 1].grid(which = 'major')
ax[1, 0].grid(which = 'major')
ax[1, 1].grid(which = 'major')
ax[0, 0].set_xscale('log')


ax[0, 0].set_ylabel('$U^+$')
ax[0, 1].set_ylabel('$\overline{uv}^+$')
ax[1, 0].set_ylabel('$\overline{u^2}^+$')
ax[1, 1].set_ylabel('$\overline{v^2}^+$')


ax[0, 0].set_xlabel('$y^+$')
ax[0, 1].set_xlabel('$y^+$')
ax[1, 0].set_xlabel('$y^+$')
ax[1, 1].set_xlabel('$y^+$')

fig.tight_layout()
plt.savefig(f'figs/{data_name}_Profiles.pdf', bbox_inches='tight')
# %%
plt.figure(0)
plt.semilogy(d["hist"][:,0],label="$L$")
plt.semilogy(d["hist"][:,1],label="$L_{b}$")
plt.semilogy(d["hist"][:,2],label="$L_{e}$")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"figs/{data_name}_Loss.pdf", bbox_inches='tight')
# %%
pred = d["pred"]
ref = d["ref"]

err = l2norm_err(ref, pred)
print(  f"Error U = {np.round(err[0],3)}%"+
        f"\nError V = {np.round(err[1],3)}%"+
        f"\nError uv = {np.round(err[2],3)}%"+
        f"\nError uu = {np.round(err[3],3)}%"+
        f"\nError vv = {np.round(err[4],3)}%"+
        f"\nError P = {np.round(err[5],3)}%"
        )

# %%
