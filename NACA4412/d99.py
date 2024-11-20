import numpy as np

def d99(yn, U):
    
    Ue = np.max(U) #0.994 # 
    idx_99 = np.argmax(U > 0.99 * Ue)
    
    d99 = yn[idx_99]
    idx = idx_99 + 1
    d1 = np.trapz(1 - U[:idx] / Ue, yn[:idx])
    d2 = np.trapz( U[:idx] / Ue * (1 - U[:idx] / Ue), yn[:idx])
    H12 = d1 / d2
    
    return np.array([idx_99, d99, Ue, d1, d2, H12])


def rep_d99(yn, U):
    n = U.shape[0]
    d = []
    for i in range(n):
        d.append(d99(yn[i], U[i]))
    
    return np.array(d)


def rep_grad(U, yn):
    n = U.shape[0]
    d = []
    for i in range(n):
        d.append(np.gradient(U[i], yn[i]))
    
    return np.array(d)