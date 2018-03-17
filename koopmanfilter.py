import numpy as np
from kalmflter import kalmflter

def koopmanfilter(n, y, trans, z, a.int, P.int, sigma, Q, R, causal.period=np.array(None), output.var.cov=False):
    
    """
    Koopman-filter, that samples alpha from posterior distribution. 

    Args:
        n: dimension of state equation
        y: observed data
        trans: coefficient matrix of state equation
        z: coefficient vector/matrix of hidden process
        a.int: starting point of mean of alpha
        P.int: starting point of variance of alpha
        sigma: variance of state equation
        Q: covariance-variance matrix of hidden process
        R: rank deficient matrix 
        causal.period: Causal period
        output.var.cov Asks to give mean a_t, variance p_t and covariance p_{t, t-1}. This is used for EMVS algorithm.

    Returns:
        alpha.sigma: sampled alpha from posterior distribution.
    """
    
    output_koopmanfilter = []

    # get data length and dimension
    T = y.shape[0]
    d = y.shape[1]
  
    # create empty matrix to collect data
    alpha.sample = np.zeros((n,T))
    v.sample = np.zeros((T, d))
    F.sample np.zeros((T, d, d)) 
    L.sample = np.zeros(n, T, n) 
    alpha.plus np.zeros(T, n)
    r = np.zeros(n, T + 1) 
    N = np.zeros(T + 1, n, n)
    a.sample = np.zeros(n, T)
    a.ff = np.zeros(n, T + 1) 
    a.ff[:,0] = a.int
    P.sample = np.zeros(T, n, n)
    P.ff = np.zeros(T + 1, n, n)
    P.ff[0, :, :] = P.int
    P.cov.sample = np.zeros(T - 1, n, n) 
    
    # kalman-filter
    if causal.period.all() is None:
        a = a.int
        P = P.int 
        for t in range(T):
            kalmanfilter = kalmflter(y[t], trans, z, a, P, sigma, Q, R)
            v.sample[t, :] = kalmanfilter[0] # v
            F.sample[:, :, t] = kalmanfilter[1] # FF
            L.sample[t, :, :] = kalmanfilter[2] # L
            a = kalmanfilter[3] #a
            P = kalmanfilter[4] #P
            a.ff[:, t + 1] = a
            P.ff[:,:, t + 1] = (P + P.T) / 2 # force to be symmetric matrix
    else:
        a = a.int
        P = P.int
        for t in range(T):
            if t in causal.period:
                kalmanfilter = kalmflter(y[t, :], trans, z, a, P, sigma, Q, R, missing=True)
                a = kalmanfilter[3] #a
                P = kalmanfilter[4] #P
                a.ff[:, t + 1] = a
                P.ff[:, :, t + 1] = P
            else:
                kalmanfilter = kalmflter(y[t, :], trans, z, a, P, sigma, Q, R)
                v.sample[t, :] = kalmanfilter[0] # v
                F.sample[:, :, t] = kalmanfilter[1] # FF
                L.sample[t, :, :] = kalmanfilter[2] # L
                a = kalmanfilter[3] #a
                P = kalmanfilter[4] #P
                a.ff[:, t + 1] = a
                P.ff[:,:, t + 1] = (P + P.T) / 2
                if causal.period.all() is not None and t == causal.period[0] - 1:
                    a.last = a
                    P.last = P 
                    
                    

    # ------------ BACKWARD RECURSION ---------------- #
    # backward recursion to obtain draws of r and N
    for t in range(T - 1, -1, -1):
        if t in causal.period:
            r[:, t] = r[:, t + 1] @ trans
            N[:, :, t] = t(trans) @ N[:, :, t + 1] @ trans
        else:
            r[:, t] = z @ np.linalg.inv(F.sample[:, :, t]) @ v.sample[t, :] + L.sample[t, :, :].T @ r[:, t + 1]
            N[:, :, t] = (z @ np.linalg.inv(F.sample[:, :, t]) @ z.T 
                          + L.sample[t, :, :].T @ N[:, :, t + 1] @ L.sample[t, :, :].T)
            
    # obtain draws of alpha
    alpha.sample[:, 0] = a.int + P.int @ r[:, 0]
    a.sample[:, 0] = a.int + P.int @ r[:, 0]
    P.sample[:, :, 0] = P.ff[:, :, 0] - P.ff[:, :, 1] @ N[:, :, 1] @ P.ff[:, :, 1]
    for t in range(1, T):
    alpha.sample[:, t] = trans @ alpha.sample[, t - 1] + R @ Q @ R.T @ r[:, t]
    a.sample[:, t] = a.ff[:, t] + P.ff[:, :, t] @ r[:, t]
    P.sample[:, :, t] = P.ff[,,t] - P.ff[:, :, t] @ N[:, :, t] @ P.ff[:, :, t]
    P.sample[:, :, t] = (P.sample[:, :, t] + P.sample[:, :, t].T) / 2 
    P.cov.sample[:, :, t - 1] = P.ff[:, :, t - 1] @ L.sample[t - 1, :, :].T @ (np.eye(n) - N[:, :, t] @ P.ff[:, :, t])
    alpha.sample = alpha.sample.T
    a.sample = a.sample
    P.sample = P.sample
    P.cov.sample = P.cov.sample
    
    if output.var.cov == True:
        output_koopmanfilter.append(alpha.sample)
        output_koopmanfilter.append(a.sample)
        output_koopmanfilter.append(P.sample)
        output_koopmanfilter.append(P.cov.sample)
        return output_koopmanfilter
    else 
    
    # return value
    if causal.period.all() is None:
        return alpha.sample
    else:
        output_koopmanfilter.append(alpha.sample)
        output_koopmanfilter(a.last)
        output_koopmanfilter(P.last)
        return output_koopmanfilter