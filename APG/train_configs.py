class APG_config:
    """
    Train configs for APG 
    Args:
        act: activation func used for MLP 
        
        n_adam: Number of steps used for supervised training
        
        n_neural: Hidden dim fo each MLP layer (N_h)
        
        n_layer: total MLP layers used in model (N_l)
        
        cp_step: length of interval to 
                    collect collection points (N_e)
        
        cb_step: length of interval to 
                    collect points on the domain boundaries (N_b) 

    """
    act = "tanh"
    n_adam = 1000
    n_neural = 20
    n_layer = 8  
    cp_step = 100
    bc_step = 10
    method = "L-BFGS-B"