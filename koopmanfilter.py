import numpy as np
from kalmflter import kalmflter

def koopmanfilter(n, y, trans, z, a_int, P_int, sigma, Q, R, causal_period=np.array(None), output_var_cov=False):
    
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
    T = y.shape[0]
    d = y.shape[1]

    # create empty matrix to collect data
    alpha_sample = np.zeros((n,T))
    v_sample = np.zeros((T, d))
    F_sample = np.zeros((d, d, T)) 
    L_sample = np.zeros((T, n, n)) 
    alpha_plus = np.zeros((T, n))
    r = np.zeros((n, T + 1)) 
    N = np.zeros((n, n, T + 1))
    a_sample = np.zeros((n, T))
    a_ff = np.zeros((n, T + 1))
    a_ff[:,0] = a_int
    P_sample = np.zeros((n, n, T))
    P_ff = np.zeros((n, n, T + 1))
    P_ff[:, :, 0]  = P_int
    P_cov_sample = np.zeros((n, n, T - 1)) 

    # kalman-filter
    if causal_period.all() is None:
        a = a_int
        P = P_int 
        for t in range(T):
            kalmanfilter = kalmflter(y[t], trans, z, a, P, sigma, Q, R)
            v_sample[t, :] = kalmanfilter['v']
            F_sample[:, :, t] = kalmanfilter['FF']
            L_sample[t, :, :] = kalmanfilter['L']
            a = kalmanfilter['a']
            P = kalmanfilter['P']
            a_ff[:, t + 1] = a
            P_ff[:,:, t + 1] = (P + P.T) / 2 # force to be symmetric matrix
    else:
        a = a_int
        P = P_int
        for t in range(T):
            if t in causal_period:
                kalmanfilter = kalmflter(y[t, :], trans, z, a, P, sigma, Q, R, missing=True)
                a = kalmanfilter['a'] 
                P = kalmanfilter['P'] 
                a_ff[:, t + 1] = a
                P_ff[:, :, t + 1] = P
            else:
                kalmanfilter = kalmflter(y[t, :], trans, z, a, P, sigma, Q, R)
                v_sample[t, :] = kalmanfilter['v'] 
                F_sample[:, :, t] = kalmanfilter['FF'] 
                L_sample[t, :, :] = kalmanfilter['L'] 
                a = kalmanfilter['a'] 
                P = kalmanfilter['P'] 
                a_ff[:, t + 1] = a
                P_ff[:,:, t + 1] = (P + P.T) / 2
                if causal_period.all() is not None and t == causal_period[0] - 1:
                    a_last = a
                    P_last = P 
                    
                    
    # backward recursion to obtain draws of r and N
    for t in range(T - 1, -1, -1):
        if t in causal_period:
            r[:, t] = r[:, t + 1] @ trans
            N[:, :, t] = trans.T @ N[:, :, t + 1] @ trans
        else:
            r[:, t] = z @ np.linalg.inv(F_sample[:, :, t]) @ v_sample[t, :] + L_sample[t, :, :].T @ r[:, t + 1]
            N[:, :, t] = (z @ np.linalg.inv(F_sample[:, :, t]) @ z.T 
                          + L_sample[t, :, :].T @ N[:, :, t + 1] @ L_sample[t, :, :].T)

        # obtain draws of alpha
    alpha_sample[:, 0] = a_int + P_int @ r[:, 0]
    a_sample[:, 0] = a_int + P_int @ r[:, 0]
    P_sample[:, :, 0] = P_ff[:, :, 0] - P_ff[:, :, 1] @ N[:, :, 1] @ P_ff[:, :, 1]
    for t in range(1, T):
        alpha_sample[:, t] = trans @ alpha_sample[:, t - 1] + R @ Q @ R.T @ r[:, t]
        a_sample[:, t] = a_ff[:, t] + P_ff[:, :, t] @ r[:, t]
        P_sample[:, :, t] = P_ff[:, :, t] - P_ff[:, :, t] @ N[:, :, t] @ P_ff[:, :, t]
        P_sample[:, :, t] = (P_sample[:, :, t] + P_sample[:, :, t].T) / 2 
        P_cov_sample[:, :, t - 1] = P_ff[:, :, t - 1] @ L_sample[t - 1, :, :].T @ (np.eye(n) - N[:, :, t] @ P_ff[:, :, t])
    alpha_sample = alpha_sample.T
    a_sample = a_sample
    P_sample = P_sample
    P_cov_sample = P_cov_sample

    if output_var_cov == True:
        
        return {"alpha sample": alpha_sample, "a sample": a_sample, "P sample": P_sample, "P cov sample": P_cov_sample}
    else: 
    # return value
        if causal_period.all() is None:
            
            return {"alpha sample": alpha_sample}
        else:
            
            return {"alpha sample": alpha_sample, "a last": a_last, "P last": P_last}