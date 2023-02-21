#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from error import l2norm_err
from train_configs import phill_config 
"""
Postprocessing of prediction data 

Return: 
    1. Loss vs Epoch
    2. U contour: Prediction, Reference, L1-norm
    3. vv contour: Prediction, Reference, L1-norm
    4. Print l2-norm error 
"""
cmp = sns.color_palette('RdBu_r', as_cmap=True)
plt.set_cmap(cmp)
#%%
act = phill_config.act
nn = phill_config.n_neural
nl = phill_config.n_layer
n_adam = phill_config.n_adam
data_name = f"phill_{nn}_{nl}_{act}_{n_adam}_2430_grid"
data = np.load(f"pred/res_{data_name}.npz")

d = np.load('data/stsphill.npz')
pos  = d["pos"]
scal = d['scal']
x = pos[0]
ind = (x[0] > 1) & (x[0] < 5)
x = pos[0][:, ind]
y = pos[1][:, ind]
u = scal[0][:, ind]
# %%
plt.figure(0)
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
names = ["U","VV"]
orders = [0,-2]
# %%
for i, name in enumerate(names):
    fig,axs = plt.subplots(3,figsize=(9,10))
    axs[0].contourf(x,y,pred[orders[i],:,:])
    axs[1].contourf(x,y,ref[orders[i],:,:])
    clb=axs[2].contourf(x,y,np.abs(ref[i,:,:]-pred[i,:,:]))
    cbar=plt.colorbar(clb,orientation="horizontal",pad = 0.3)
    axs[0].set_title(f"{name} Prediction",fontdict={"size":18})
    axs[1].set_title("Reference",fontdict={"size":18})
    axs[2].set_title("Error",fontdict={"size":18})
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"figs/{data_name}_{name}",dpi= 300)

# %%
err = l2norm_err(ref, pred)
print(  f"Error U = {np.round(err[0],3)}%"+
        f"\nError V = {np.round(err[1],3)}%"+
        f"\nError uv = {np.round(err[2],3)}%"+
        f"\nError uu = {np.round(err[3],3)}%"+
        f"\nError vv = {np.round(err[4],3)}%"+
        f"\nError P = {np.round(err[5],3)}%"
        )
