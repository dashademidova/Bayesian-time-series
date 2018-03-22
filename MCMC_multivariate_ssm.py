# By Bo Ning
# Mar. 6. 2017

######################### Model ################################
## Observation equation:                                      ##
##         Y_t = mu_t + delta_t + epsilon_t ~ N(0, sigma)     ##
## Hidden state:                                              ##
##         mu_(t+1) = mu_t + tau_t + u_t ~ N(0, sigma.U)      ##
##         tau_(t+1) = tau_t + v_t ~ N(0, sigma.V)            ##
## (Nonstationary)                                            ##
##         tau_(t+1) = D + phi %*% (tau_t - D) + v_t          ##
##         delta_(t+1) = - sum_{j=0}^{S-2} delta_(t-s) + w_t  ##
##                            w_t ~ N(0, sigma.W)             ##
##                                                            ##
## ------------------- State-space Model -------------------- ##
## Observation equation:                                      ##
##          Y_t = z %*% alpha_t + epsilon_t                   ##
## State equation:                                            ##
##         alpha_(t+1) = c + T %*% alpha_t + R %*% Q          ##
##              Q ~ N(0, bdiag(sigma.U, sigma.V, sigma.W))    ##
################################################################

#' @param test_dataset: T by n dataset with time period T, number of datasets
#'         n
#' @param causal.period: Period of dataset has causal impact
#' @param nseasons: seasonality input for analysis
#' @param iterloop: iterations for MCMC
#' @param burnin: burn-in
#' @param stationary: adding constrain hidden state to be stationary or not
#' @param graph: impose graph restriction on the variance covariance matrix
#' @param graph.structural: if graph is TRUE, specify graphic structure 
#'        matrix
import numpy as np
import numpy.linalg as la
import scipy
import scipy.stats
from scipy.stats import wishart, multivariate_normal
import scipy.linalg as sla
from statsmodels.tsa.api import VAR
from koopmanfilter import koopmanfilter
from tqdm import tqdm
from varp import sqrtm

path = "../Bayesian-multivariate-time-series-causal-inference/BayesianCausalInference/"

def chol2inv(x):
    inv_local = np.linalg.inv(np.triu(x))
    return inv_local.dot(inv_local.T)

def check_positive(A):
    u, s, _ = la.svd(A)
    return u.dot(np.diag(s).dot(u.T))

def MCMC_multivariate_ssm(test_data, causal_period, nseasons=12, iterloop = 1000, burnin=100, 
                          stationary=True, graph=False, graph_structure=None, seed=3):

    
    ############## load functions ###############
    if stationary:
        from stationaryRestrict import stationaryRestrict
    
    ############### organize dataset #################
    length, d = test_data.shape
    # seperate causal period dataset
    causal_period = causal_period
    causal_period_len = causal_period.shape[0]
    length_non_causal = length - causal_period_len
    
    ############### initialize parameters #################
    # initialize z
    circle = np.min((nseasons, length))
    n = (circle+1)*d
    z = np.zeros((n, d))
    z[:d, :] = np.eye(d)
    z[2*d:3*d, :] = np.eye(d)
    
    # initialize mu: intercept of hidden equation
    mu_ss = np.zeros(n)
    
    # initialize alpha: (see Durbin and Koopman, 2002)
    alpha = np.zeros(n, dtype=np.int32)
    aStar = np.zeros(n, dtype=np.int32) 
    # initialize variance of alpha
    P = np.zeros((n, n), dtype=np.int32)
    P[:3*d, :3*d] = np.eye(3*d) * 1e6
    if stationary:
        P[d:2*d, d:2*d] = np.eye(d)
    
    # initialize sigma
    delta = d+1 # prior for sigma
    B = np.eye(d) # prior for sigma
    # give right parameters t0 wishart
    sigma_hat_inv = wishart(df=d+1, scale=np.eye(d)/.1**2/d).rvs() * graph_structure

    # what's happening here
    sigma_hat = chol2inv(sigma_hat_inv)
    
    # initialize transition matrix
    trans = np.zeros((n, n))
    linear = np.eye(2*d)
    linear[:d, d:2*d] = np.eye(d)
    trans[:2*d, :2*d] = linear
    # take initial variance of tau from the data
    if stationary:
        # find python library with autoregressive models
        var = VAR(test_data[causal_period, :])
        data_phi = var.fit(maxlags=1, ic=None, trend="nc").params
        trans[d:2*d, d:2*d] = data_phi
    else:
        trans[d:2*d, d:2*d] = np.eye(d)

    seasonal = np.zeros((circle-1, circle-1))
    seasonal[0, :] = -1
    seasonal[1:circle-1, :circle-2] = np.eye(circle-2)
    for dims in range(d):
        trans[2*d+dims: n: d, 2*d+dims: n: d] = seasonal
    
    # initialize R 
    R = np.zeros((n, d*3))
    R[:3*d, :3*d] = np.eye(3*d)
    
    # initialize covariance matrix Q = bdiag(sigmaU, sigmaV, sigmaW)
    k1, k2, k3 = 0.1, 0.1, 0.1
    # what's happening here
    if not graph:
        sigmaU = chol2inv(wishart(d, k1**2*d*np.eye(d)).rvs())
        sigmaV_inv = wishart(d, k2**2*d*np.eye(d)).rvs()
        sigmaV = chol2inv(sigmaV_inv)
        sigmaW = chol2inv(wishart(d, k3**2*d*np.eye(d)).rvs())
    else:
        sigmaU = chol2inv(wishart(df=d+1, scale=np.eye(d)/k1**2/d).rvs() * graph_structure)
        sigmaV_inv = wishart(df=d+1, scale=np.eye(d)/k1**2/d).rvs() * graph_structure
        sigmaV = chol2inv(sigmaV_inv)
        sigmaW = chol2inv(wishart(df=d+1, scale=np.eye(d)/k1**2/d).rvs() * graph_structure)

    Q = sla.block_diag(sigmaU, sigmaV, sigmaW)
    
    ################### Prepare for MCMC Sampling ##################
    # create matrix to store parameter draws
    mu_sample = np.empty((length, d, iterloop))
    a_last_sample = np.empty((n, iterloop))
    P_last_sample = np.empty((n, n, iterloop))
    prediction_sample = np.empty((length, d, iterloop))
    sigma_sample = np.empty((d, d, iterloop))
    sigma_U_sample = np.empty((d, d, iterloop))
    sigma_V_sample = np.empty((d, d, iterloop))
    sigma_W_sample = np.empty((d, d, iterloop))
    if stationary:
        Theta_sample = np.empty((d, d, iterloop))
        D_sample = np.zeros((d, iterloop))

    # pb  = txtProgressBar(1, iterloop, style=3)    # report progress
    # print("\nStarting MCMC sampling: \n")     # report progress
    ##################### Begin MCMC Sampling #######################
    # ptm = proc.time()
    for itery in tqdm(range(iterloop), desc="MCMC sampling"):
        ## --------------------------------------- ##
        ## Step 1. obtain draws of alpha, apply Koopman's filter (2002)
        # simulate w.hat, y.hat, alpha.hat for Koopman's filter (2002)
        alpha_plus = np.zeros((length, n))
        for t in range(length):
            eta = multivariate_normal(mean=np.zeros(3*d), cov=Q, seed=seed).rvs()
            if t == 0:
                alpha_plus[t, :] = mu_ss + trans.dot(alpha) + R.dot(eta)
            else:
                alpha_plus[t, :] = mu_ss + trans.dot(alpha_plus[t-1, :]) + R.dot(eta)

        test_est_plus = alpha_plus.dot(z) + multivariate_normal(mean=np.zeros(d), 
                                                                cov=sigma_hat, 
                                                                seed=seed).rvs(size=length)
        test_est_star = test_data - test_est_plus 
        # Estimate alpha parameters
        sample_alpha_draws = koopmanfilter(n, test_est_star, trans, z, aStar, 2*P, 2*sigma_hat, 2*Q, R, causal_period)
        alpha_star_hat = sample_alpha_draws["alpha sample"]
        alpha_draws = alpha_star_hat + alpha_plus
      
        # collect a.last and P.last, 
        # use them for starting point of koopman filter for causal period dataset
        # print(a_last_sample.shape, sample_alpha_draws["a last"].shape)
        a_last_sample[:, itery] = sample_alpha_draws["a last"]
        P_last_sample[:, :, itery] = sample_alpha_draws["P last"]
      
        ## ---------------------------------------- ##
        ## Step 2: make stationary restriction
        if stationary:
            alpha_draws_tau = alpha_draws[:length_non_causal, d:d*2]
            if itery == 0:
                alpha_draws_tau_demean = alpha_draws_tau
                Theta_draw = stationaryRestrict(alpha_draws_tau_demean, sigmaV, sigmaV_inv)
            else:
                alpha_draws_tau_demean = (alpha_draws_tau.T - D_draw.reshape(-1, 1)).T
                Theta_draw = stationaryRestrict(alpha_draws_tau_demean, sigmaV_draws, sigmaV_inv)
            trans[d:2*d, d:2*d] = Theta_draw
        
            ## ---------------------------------------- ##
            ## Step 3: sample intercept mu.D, denote N(0, I) prior for D
            tau_part_A = alpha_draws_tau[1:length_non_causal, :] \
                         - alpha_draws_tau[:length_non_causal-1, :].dot(Theta_draw.T)
            tau_part_B = np.eye(d) - Theta_draw
            D_var = la.inv((length_non_causal-1)*tau_part_B.T.dot(sigmaV_inv).dot(tau_part_B) + np.eye(d))
            D_var = check_positive(D_var)
            D_mean = D_var.dot((tau_part_B.T.dot(sigmaV_inv)).dot(tau_part_A.sum(axis=0)))#.reshape(-1, 1)))  #it's wrong
            D_draw = multivariate_normal(mean=D_mean, cov=D_var, seed=seed).rvs()
            # update the mean: D - theta * D
            D_mu = tau_part_B.dot(D_draw)
            # print(D_mu.shape, mu_ss.shape)
            mu_ss[d:2*d] = D_mu
            # update alpha_draws_tau_demean
            alpha_draws_tau_demean = (alpha_draws_tau.T - D_draw.reshape(-1, 1)).T
      
        ## ---------------------------------------- ##
        ## Step 4: update sigmaU, sigmaV, sigmaW
        # parameter in sigmaU
        PhiU_value = alpha_draws[1:length_non_causal, :d] \
                     - alpha_draws[:length_non_causal-1, :d] \
                     - alpha_draws[:length_non_causal-1, d:d*2]

        PhiU = PhiU_value.T.dot(PhiU_value)
        # parameter in sigmaV
        if stationary:
            PhiV_value = alpha_draws_tau_demean[1:length_non_causal, :] \
                         - alpha_draws_tau_demean[:length_non_causal-1, :].dot(Theta_draw.T)
            PhiV = PhiV_value.T.dot(PhiV_value)
        else:
            PhiV_value = alpha_draws[1:length_non_causal, d:2*d] \
                         - alpha_draws[:length_non_causal-1, d:2*d]
            PhiV = PhiV_value.T.dot(PhiV_value)
        # parameter in sigmaW
        bind_W = np.zeros((causal_period[0]-1, 0))
        for dims in range(d):
            bind_W_tmp = np.hstack((alpha_draws[1:length_non_causal, d*2+dims:n:d],
                                    alpha_draws[:length_non_causal-1, n-d+dims, None]))
            bind_W = np.hstack((bind_W, bind_W_tmp.sum(axis=1).reshape(-1, 1)))

        PhiW = bind_W.T.dot(bind_W)

        scale_U = PhiU + (d+1)*k1**2*np.eye(d)
        scale_V = PhiV + (d+1)*k2**2*np.eye(d)
        scale_W = PhiW + (d+1)*k3**2*np.eye(d)

        # start from here
        # sample sigmaU, sigmaV, sigma W from their posteriors
        if not graph:
            sigmaU_draws = la.inv(wishart(length_non_causal+d-1, scale_U).rvs())
            sigmaV_inv = wishart(length_non_causal+d-1, scale_V).rvs()
            sigmaV_draws = la.inv(sigmaV_inv)
            sigmaW_draws = la.inv(wishart(length_non_causal+d-1, scale_W).rvs())
        else:
            sigmaU_inv = wishart(length_non_causal+d-1, la.inv(scale_U)).rvs() * graph_structure
            sigmaU_draws = la.inv(check_positive(sigmaU_inv))
            sigmaV_inv = wishart(length_non_causal+d-1, la.inv(scale_V)).rvs() * graph_structure
            sigmaV_draws = la.inv(check_positive(sigmaV_inv))
            sigmaW_inv = wishart(length_non_causal+d-1, la.inv(scale_W)).rvs() * graph_structure
            sigmaW_draws = la.inv(check_positive(sigmaW_inv))
        
        Q = sla.block_diag(sigmaU_draws, sigmaV_draws, sigmaW_draws)
      
        ## ---------------------------------------- ##
        ## Step 5: update sigma_hat
        res = (test_data - alpha_draws.dot(z))[:length_non_causal, :]
        if not graph:
            D_sigma = res.T.dot(res) + B
            sigma_hat_inv = wishart(delta+length_non_causal, D_sigma).rvs()
            sigma_hat = la.inv(check_positive(sigma_hat_inv))
        else:
            D_sigma = res.T.dot(res) + B
            sigma_hat_inv = wishart(delta+length_non_causal, la.inv(D_sigma)).rvs() * graph_structure
            sigma_hat = la.inv(check_positive(sigma_hat_inv))
      
        ## ---------------------------------------- ##
        ## Step 6: estimating dataset using predicted value
        prediction_sample[:, :, itery] = alpha_draws.dot(z) + multivariate_normal(mean=np.zeros(d), 
                                                                                  cov=sigma_hat,
                                                                                  seed=seed).rvs(alpha_draws.shape[0])
        ## ---------------------------------------- ##
        ## Step 7: collect sample draws
        mu_sample[:, :, itery] = alpha_draws[:length, :d]
        if stationary:
            Theta_sample[:, :, itery] = Theta_draw
            D_sample[:, itery] = D_draw
        sigma_sample[:, :, itery] = sigma_hat
        sigma_U_sample[:, :, itery] = sigmaU_draws
        sigma_V_sample[:, :, itery] = sigmaV_draws
        sigma_W_sample[:, :, itery] = sigmaW_draws
    
    # return result
    if stationary:
        return_dict  = {"prediction sample": prediction_sample, 
                        "mu sample": mu_sample,
                        "Theta sample": Theta_sample, 
                        "D sample": D_sample, 
                        "sigma sample": sigma_sample, 
                        "sigma U sample": sigma_U_sample,
                        "sigma V sample": sigma_V_sample, 
                        "sigma W sample": sigma_W_sample,
                        "a last sample": a_last_sample, 
                        "P last sample": P_last_sample,
                        "z": z, "R": R, "trans": trans}
    else:
        return_dict  = {"prediction sample": prediction_sample, 
                        "mu sample": mu_sample,
                        "sigma sample": sigma_sample, 
                        "sigma U sample": sigma_U_sample,
                        "sigma V sample": sigma_V_sample, 
                        "sigma W sample": sigma_W_sample,
                        "a last sample": a_last_sample, 
                        "P last sample": P_last_sample,
                        "z": z, "R": R, "trans": trans}

    return return_dict
