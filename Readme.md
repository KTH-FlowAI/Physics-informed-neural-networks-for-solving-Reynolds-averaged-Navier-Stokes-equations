# Physics-informed neural networks for solving Reynolds-averaged Navier–Stokes equations

## Introduction
The code in this repository features a Python implemention of Physics-informed neural networks (PINNs) for solving the Reynolds-averaged Navier–Stokes (RANs)equations for incompressible turbulent flows without any specific model or assumption for turbulence. The input data are only taken at the domain boundaries. 

More details about the implementation and results from the training are available in ["Physics-informed neural networks for solving Reynolds-averaged Navier–Stokes equations", Hamidreza Eivazi, Mojtaba Tahani, Philipp Schlatter, Ricardo Vinuesa](https://aip.scitation.org/doi/abs/10.1063/5.0095270)(2022,*Physics of Fluids*)

## Pre-requisites
The code was run successfully using Tensorflow>=2.6.0, using 1 GPU for training. In addition, scipy is necessary for implementing optimization algorithm

## Data
The dataset used for training and testing are available in order to ensure the reproducibility of the results. 
Now we only offer dataset for *Falker-Skan* Boundary Layer (**/FS**) and adverse-pressure-gradient (APG) Boundary Layer (**/APG**) case.
Please, get in touch using the email address for correspondance in the paper to arrange the transfer. 

## 

