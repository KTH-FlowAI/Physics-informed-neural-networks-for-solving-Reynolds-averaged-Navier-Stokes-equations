# Physics-informed neural networks for solving Reynolds-averaged Navier–Stokes equations

## Introduction
The code in this repository features a Python implemention of Physics-informed neural networks (PINNs) for solving the Reynolds-averaged Navier–Stokes (RANs)equations for incompressible turbulent flows without any specific model or assumption for turbulence. The input data are spatial coordinates (*x*, *y*) only taken at the domain boundaries. 

More details about the implementation and results from the training are available in ["Physics-informed neural networks for solving Reynolds-averaged Navier–Stokes equations", Hamidreza Eivazi, Mojtaba Tahani, Philipp Schlatter, Ricardo Vinuesa](https://aip.scitation.org/doi/abs/10.1063/5.0095270)(2022,*Physics of Fluids*)

## Pre-requisites
The code was run successfully using Tensorflow>=2.6.0, using 1 GPU for training. In addition, scipy is necessary for implementing optimization algorithm

## Data
The dataset used for training and testing are available in order to ensure the reproducibility of the results. 

Now we only offer dataset for *Falker-Skan* Boundary Layer [**FS**](https://github.com/Fantasy98/Physics-informed-neural-networks-for-solving-Reynolds-averaged-Navier-Stokes-equations/blob/9f05eac466ed6f15638de9ec173e4826059b0b49/FS/data/FalknerSkan_n0.08.npz) and adverse-pressure-gradient (APG) Boundary Layer [**APG**](https://github.com/Fantasy98/Physics-informed-neural-networks-for-solving-Reynolds-averaged-Navier-Stokes-equations/blob/9f05eac466ed6f15638de9ec173e4826059b0b49/APG/data/APG_b1n.npz) case.

Please, get in touch using the email address for correspondance in the paper to arrange the transfer. 

##  Training and inference
The PINNs training and prediction can be performed after cloning the repository. 

Take *Falker-Skan*(FS) Boundary Layer as example:
    
    git clone  https://github.com/Fantasy98/Physics-informed-neural-networks-for-solving-Reynolds-averaged-Navier-Stokes-equations.git
    cd FS
    python train.py


All the training parameters are defined in the [train_configs.py](https://github.com/Fantasy98/Physics-informed-neural-networks-for-solving-Reynolds-averaged-Navier-Stokes-equations/blob/9f05eac466ed6f15638de9ec173e4826059b0b49/FS/train_configs.py)

For postprocessing the reults, it can be performed as follows:
    
    python postprocessing.py

## File Structure
For *Falker-Skan* (**FS**) and adverse-pressure-gradient (**APG**) Boundary Layer case, take *Falker-Skan*(FS) Boundary Layer as example, the files follows structure below:

    FS
    ├── data
    ├── pred
    ├── figs
    ├── models
    ├── train_configs.py
    ├── PINN.py
    ├── lbfgs.py
    ├── postprocessing.py
    ├── error.py
    └── train.py

***data***: Folder to store the boundary layer data
***pred***: Folder to store the prediction results
***models***: Folder to store the trainede models    
***figs***: Folder to store the postprocessed figures     
***train_configs.py***: Definition of training parameters   
***PINN.py***: class object of PINN
***lbfgs.py***: optimizer based on L-BFGS algorithm
***postprocessing.py***: script for postprocessing results. 
***error.py***: function for error assessment used in paper.
***train.py***: script for training PINNs model 



