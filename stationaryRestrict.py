# Fitting a Stationary VAR(1) model for tau_t
from varp import * 
from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np
def stationaryRestrict(y, sigma, sigma_inv=[None]):
    m = sigma.shape[0]
    priorsig = np.array([5] * m**2)
    jumpsig = np.array([0.05] * m**2)
    priorlogor = 0 
    jump = 0.5
    y = y.T
    length = y.shape[1]
    if sigma_inv.all() is None:
            sigma_inv = np.linalg.inv(sigma)
    ARModel = VAR(y.T)
    phi = VAR.fit(ARModel, 1, 'ols', None, 'nc', True).params.T
    phi = initial_var(phi, Shrink = 1)
    pre = par2pre_varp(phi)['pre']
    U = par2pre_varp(phi)['U']
    delta = par2pre_varp(phi)['delta']
    
    # 
    if np.linalg.det(U - np.eye(m)) <= 1e-10:
        phi = phi
    else:
        length_v = int(m * (m + 1) / 2)
        prenew = pre
        prenew[0 : length_v - 1] = pre[0 : length_v - 1] + np.random.normal(0, jumpsig[0], length_v - 1)
        logr = (-0.5 * np.sum((prenew[0 : length_v - 1]**2 - pre[0 : length_v - 1]**2) 
                              / (priorsig[0 : length_v - 1]**2)) - varppre_lkhd(y, prenew, delta, sigma, sigma_inv) 
                + varppre_lkhd(y, pre, delta, sigma, sigma_inv))
        if np.log(np.random.random(1)) < logr:
            pre[0 : length_v - 1] = prenew[ 0 : length_v - 1] 

        # update q, given v and delta    
        prenew[length_v : (m**2) - 1] = (pre[length_v : m**2 - 1] 
                                         + np.random.normal(0, jumpsig[0], m**2 - length_v - 1))
        logr = (-0.5 * np.sum((prenew[length_v : m**2 - 1]**2 
                               - pre[length_v : m**2 - 1]**2) / (priorsig[length_v : m**2 - 1]**2)) - 
                varppre_lkhd(y, prenew, delta, sigma, sigma_inv) +
                varppre_lkhd(y, pre, delta, sigma, sigma_inv))
        if np.log(np.random.random(1)) < logr:
            pre[length_v : m**2 - 1] = prenew[length_v : m**2 - 1]
        # update delta given v, q
        deltanew = delta
        deltanew = 2 * np.random.binomial(1, jump, 1)[0] - 1    
        logr = (np.sum(np.sign(deltanew-delta) * priorlogor) - varppre_lkhd(y, pre, deltanew, sigma, sigma_inv) 
                + varppre_lkhd(y, pre, delta, sigma, sigma_inv))
        if np.log(np.random.random(1)) < logr:
            delta = deltanew
        phi = pre2par_varp(pre, delta)

    return phi