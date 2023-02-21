#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from error import l2norm_err
from train_configs import NACA_config
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
act = NACA_config.act
nn = NACA_config.n_neural
nl = NACA_config.n_layer
n_adam = NACA_config.n_adam
data_name = f"NACA_{nn}_{nl}_{act}_{n_adam}_2550_620"
data = np.load(f"pred/res_{data_name}.npz")
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
#%%
n = int(u.shape[1]/2)
orders= [0,2,3,4]
names = [r"${U^{+}}$",r"${\overline{uv}}^{+}$",r"${\overline{u^2}}^{+}$",r"${\overline{v^2}}^{+}$"]
file_names = ["U","uv","uu","vv"]
for i, name in enumerate(names):
    plt.figure(i+2,figsize=(8,6))
    plt.plot(y[:,n],pred[orders[i],:,n],label = "Prediction")
    plt.plot(y[:,n],ref[orders[i],:,n],label ="Reference" )
    plt.title(f"{name}",fontdict={"size":18})
    plt.xlabel(r"${y^{+}}$",fontdict={"size":16})
    plt.ylabel(name,fontdict={"size":16})
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.savefig(f"figs/{data_name}_{file_names[i]}",dpi= 300)
# %%
err = l2norm_err(ref, pred)
print(  f"Error U = {np.round(err[0],3)}%"+
        f"\nError V = {np.round(err[1],3)}%"+
        f"\nError uv = {np.round(err[2],3)}%"+
        f"\nError uu = {np.round(err[3],3)}%"+
        f"\nError vv = {np.round(err[4],3)}%"+
        f"\nError P = {np.round(err[5],3)}%"
        )

# %%
