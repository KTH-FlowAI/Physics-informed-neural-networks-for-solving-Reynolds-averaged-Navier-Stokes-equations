class ZPG_config:
    """
    Train configs for ZPG BL 
    Args:
        act: activation func used for MLP 
        
        n_adam: Number of steps used for supervised training
        
        n_neural: Hidden dim fo each MLP layer (N_h)
        
        n_layer: total MLP layers used in model (N_l)

        cp_step: length of interval to 
                    collect collection points (N_e)
        
        method: Optimizer for unsupervised training
    """
    act = "tanh"
    n_adam = 1000
    n_neural = 20
    n_layer = 8 
    cp_step = 500
    method = "L-BFGS-B"
