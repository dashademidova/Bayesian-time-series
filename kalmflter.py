import numpy as np
import sys
from MungeMatrix import MungeMatrix

def kalmflter(y, trans, z, a, P, sigma, Q, R, missing=False):
    
    """
    Kalman-filter for forward draws of alphas. 

    Args:
        y: Observed data.
        trans: Coefficient matrix of state equation.
        z: Coefficient vector/matrix of alpha.
        a: Starting point of mean of alpha.
        P: Starting point of variance of alpha.
        sigma: Variance of state equation.
        Q: Covariance-variance matrix of hidden process.
        R: Rank deficient matrix. 
        missing: Lets kalman-filter do data imputation if missing is true.

    Returns:
        output_kalmflter: Updated a, updated P, updated alpha.
    """
    if missing == True:
        a_update = trans @ a
        P_update = trans @ P @ trans.T + R @ Q @ R.T

        return {"a": a_update, "P": P_update}

    else:
        v = y - z.T @ a
        FF = z.T @ P @ z + sigma
        if np.linalg.cond(FF) < 1 / sys.float_info.epsilon:
            FF = MungeMatrix(FF)
        K = trans @ P @ z @ np.linalg.inv(FF)
        L = trans - K @ z.T
        a_update = trans @ a + K @ v
        P_update = trans @ P @ L.T + R @ Q @ R.T

        return {"v": v, "FF": FF, "L": L, "a": a_update, "P": P_update}