# NACA4412 airfoil 
PINNs for simulation of the turbulent boundary layer developing on the suction side of a NACA4412 airfoil at the Reynolds number based on 
$U_{\infty}$ and chord length $c$ of $Re_{c} = 200{,}000$. 
For more details, please check the reference [R. Vinuesa, P. S. Negi, M. Atzori, A. Hanifi, D. S. Henningson, and P. Schlatter, “Turbulent boundary layers around wing sections up to $Re_c = 1 \ 000 \ 000$,” Int. J. Heat Fluid Flow 72, 86–99 (2018).](https://doi.org/10.1016/j.ijheatfluidflow.2018.04.017)

# Get started: Reconnecting Dataset (Updated 2024.11.21)

We provide the split large datasets in [data](./data/). To reconnect the split files, please do: 

    ./gen-Wing-Data

Then one should find the dataset named [data/top2n_HighRes.npz](./data/top2n_HighRes.npz) in [data](./data/)

