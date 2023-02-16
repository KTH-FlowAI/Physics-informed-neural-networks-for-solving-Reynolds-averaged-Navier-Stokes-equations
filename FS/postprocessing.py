#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from error import l2norm_err
from train_configs import FS_config
"""
Postprocessing of prediction data 

Return: 
    1. Loss vs Epoch
    2. U contour: Prediction, Reference, L1-norm
    3. V contour: Prediction, Reference, L1-norm
    4. P contour: Prediction, Reference, L1-norm
    5. Print l2-norm error 
"""
cmp = sns.color_palette('RdBu_r', as_cmap=True)
plt.set_cmap(cmp)
#%%
act = FS_config.act
nn = FS_config.n_neural
nl = FS_config.n_layer
n_adam = FS_config.n_adam
data_name = f"FS_{nn}_{nl}_{act}_{n_adam}_1206_500"
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
# %%
for i, name in enumerate(names):
    fig,axs = plt.subplots(3,figsize=(9,10))
    axs[0].contourf(x,y,pred[i,:,:])
    axs[1].contourf(x,y,ref[i,:,:])
    clb=axs[2].contourf(x,y,np.abs(ref[i,:,:]-pred[i,:,:]))
    cbar=plt.colorbar(clb,orientation="horizontal",pad = 0.3)
    axs[0].set_title("Pressure Prediction",fontdict={"size":18})
    axs[1].set_title("Reference",fontdict={"size":18})
    axs[2].set_title("Error",fontdict={"size":18})
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"figs/{data_name}_{name}",dpi= 300)
# %%
err = l2norm_err(ref, pred)
print(f"Error U = {np.round(err[0],3)}%\nError V = {np.round(err[1],3)}%\nError P = {np.round(err[2],3)}%")
# %%
