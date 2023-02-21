class phill_config:
    """
    Train configs for Falkner-Skan BL 
    Args:
        act: activation func used for MLP 
        
        n_adam: Number of steps used for supervised training
        
        n_neural: Hidden dim fo each MLP layer (N_h)
        
        n_layer: total MLP layers used in model (N_l)
         
        method: Optimizer for unsupervised training
    """
    act = "tanh"
    
    n_adam = 1000
    n_neural = 20
    n_layer = 8  
    method = "L-BFGS-B"
