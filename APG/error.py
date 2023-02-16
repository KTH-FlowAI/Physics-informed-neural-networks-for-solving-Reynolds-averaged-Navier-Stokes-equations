import numpy as np


def l2norm_err(ref, pred):
    """
    relative l2-norm of errors E_i on all computational points for i-th variable
    
    Args:
        ref: (nparray) Reference value of variables 

        pred: (nparray) Prediction vallue computed by Neural Networks

    Return: (nparray) An array with shape of [N,I]
            N = number of points, I = number of variables

    """
    return np.linalg.norm(ref - pred, axis = (1, 2)) / np.linalg.norm(ref, axis = (1, 2)) * 100
#