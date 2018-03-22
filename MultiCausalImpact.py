# By Bo Ning
# Mar. 6. 2017

#' \code{MultiCausalImpact} the function to calculate KS distance and threshold
#' 
#' @param test.dataset: T by n dataset with time period T, number of datasets
#'         n
#' @param causal.period: Period of dataset has causal impact
#' @param nseasons: seasonality input for analysis
#' @param iterloop: iterations for MCMC
#' @param burnin: burn-in
#' @param stationary: adding constrain hidden state to be stationary or not
#' @param graph: impose graph restriction on the variance covariance matrix
#' @param graph.structural: if graph is TRUE, specify graphic structure 
#'        matrix
#' @param num.sim.data: number of counterfactual data to be simulated
#' @param probs: the percentile for deciding the threshold
#' @param num.cores: number of cores to run programming for parallel computing,
#'        the default is 1
import numpy as np
from numpy.random import seed, choice
import scipy
import scipy.linalg as sla
import scipy.stats
from scipy.stats import ks_2samp, multivariate_normal
from MCMC_multivariate_ssm import MCMC_multivariate_ssm
from koopmanfilter import koopmanfilter
from tqdm import tqdm

def MultiCausalImpact(test_data, causal_period, cntl_term, seed_local=1, nseasons=12, 
                      iterloop=1000, burnin=100, stationary=True, graph=False, 
                      graph_structure=None, num_sim_data=30, probs=0.95):
    ############## load packages #################

    seed(seed_local)
    length, d = test_data.shape

    # seperate causal period dataset
    causal_length = causal_period.shape[0] #causal_period.size
    length_non_causal = length - causal_length
    # Fit deduct cntl.term from test.data
    test_data_tilde = test_data - cntl_term
  
    #################################################
    # Step 1: Sample draws from posterior distributions of parameters
    #         and obtain the predicted distribution for causal period
    mcmc_model_output = \
        MCMC_multivariate_ssm(test_data_tilde, causal_period,
                              nseasons=nseasons, iterloop=iterloop, 
                              burnin=burnin, stationary=stationary,
                              graph=graph, graph_structure=graph_structure)
    prediction_sample = mcmc_model_output["prediction sample"]
    a_last_sample = mcmc_model_output["a last sample"]
    P_last_sample = mcmc_model_output["P last sample"]
    if stationary:
        Theta_sample = mcmc_model_output["Theta sample"]
    sigma_sample = mcmc_model_output["sigma sample"]
    sigma_U_sample = mcmc_model_output["sigma U sample"]
    sigma_V_sample = mcmc_model_output["sigma V sample"]
    sigma_W_sample = mcmc_model_output["sigma W sample"]
    D_sample =  mcmc_model_output["D sample"]
    z = mcmc_model_output["z"]
    R = mcmc_model_output["R"]
    trans = mcmc_model_output["trans"]
  
    ####################################################
    # print("\nEstimating trend for each simulated counterfactual:")
    # report progress
  
    # Step 2: Sample num.sim.data number of counterfactuals from predicted data
    num_sim_data = num_sim_data # numbers of dataset to simulate
    # generate random numbers
    counterfactual_data = np.empty((causal_length, d, num_sim_data))
    for t in range(causal_length):
        for dd in range(d):
            indices = choice(np.arange(burnin, iterloop), size=num_sim_data, replace=True) #does it really replace values?
            counterfactual_data[t, dd, :] = prediction_sample[causal_period[0]+t, dd, indices] #maybe I shouldn't remove -1 (it was +t-1)

    for num in range(num_sim_data):
        counterfactual_data[:, :, num] +=  cntl_term[causal_period, :]
  
    ####################################################
    ## Step 3:
    # combine counterfactual dataset and observed dataset
    # and fit them into the model to obtain draws of trend
    combined_data = np.dstack((counterfactual_data, test_data[causal_period, :, None])) #maybe I'm wrong -> hstack
  
    ####################################################
    ## Step 4: 
    # Fit dataset to calculate trend using parallel computing
    combined_data_estimate_trend = np.empty((causal_length, d, iterloop-burnin, num_sim_data+1))
    # Check if here right margins
    for k in tqdm(range(num_sim_data+1), desc="Fit dataset to calculate trend"):  
        data_estimate_trend = []
        for x in range(burnin, iterloop):
            if stationary:
                trans[d:2*d, d:2*d] = Theta_sample[:, :, x]
                alpha_plus = np.zeros((causal_length, (np.min((nseasons, length))+1)*d))
                Q = sla.block_diag(sigma_U_sample[:, :, x], sigma_V_sample[:, :, x], sigma_W_sample[:, :, x])
                mu_ss = np.zeros((np.min((nseasons, length))+1)*d)
                for t in range(causal_length):
                    eta = multivariate_normal(mean=np.zeros(3*d), cov=Q, seed=seed_local).rvs() 
                    mu_ss[d: 2*d] = (np.eye(d) - Theta_sample[:, :, x]).dot(D_sample[:, x])
                    if t == 0:
                        alpha_plus[t, :] = mu_ss + trans.dot(a_last_sample[:, x]) + R.dot(eta)
                    else:
                        alpha_plus[t, :] = mu_ss + trans.dot(alpha_plus[t-1, :]) + R.dot(eta)

            data_est_plus = alpha_plus.dot(z) + multivariate_normal(mean=np.zeros(d), 
                                                                    cov=sigma_sample[:, :, x],
                                                                    seed=seed_local).rvs(size=causal_length)
            data_est_star = combined_data[:, :, k] - data_est_plus 
            # Estimate alpha parameters
            sample_alpha_draws = \
                koopmanfilter((np.min((nseasons, length))+1)*d,
                              data_est_star, trans, z, a_last_sample[:, x],
                              P_last_sample[:, :, x], 2*sigma_sample[:, :, x], 2*Q, R)["alpha sample"]
            sample_alpha = sample_alpha_draws + alpha_plus
            data_estimate_trend.append(sample_alpha)

        # convert list object to array
        for i in range(iterloop - burnin):
            combined_data_estimate_trend[:, :, i, k] = np.asarray(data_estimate_trend[i][:,:d])
  
    ####################################################
    # Step 5:
    # compare two distributions:
    # \sum_T+1: T+n \mu_t | Y_obs and \sum_T+1: T+n \mu_t | Y_cf
    combined_data_estimate_culmulate_trend = \
        np.empty((causal_length, d, iterloop-burnin, num_sim_data+1))
    for t in range(causal_length):
        if t == 0:
            combined_data_estimate_culmulate_trend[t, :, :, :] = combined_data_estimate_trend[0, :, :, :]
        else:
            combined_data_estimate_culmulate_trend[t, :, :, :] = combined_data_estimate_trend[:t, :, :, :].mean(axis=0)
  
    print("Calculating ks distance...") # report progress
    ####################################################
    # Step 6: calculate the threshold for control variables
    ks_cntlsets = np.empty((causal_length, num_sim_data*(num_sim_data-1), d))
    for t in tqdm(range(causal_length), desc="Causal period"):
        for dd in range(d):
            a = 0
            for i in range(num_sim_data):
                for j in range(num_sim_data):
                    if i != j:
                        ks_cntlsets[t, a, dd] = ks_2samp(combined_data_estimate_culmulate_trend[t, dd, :, i], 
                                                         combined_data_estimate_culmulate_trend[t, dd, :, j])[0]
                        a += 1

    threshold = np.percentile(ks_cntlsets, 100 * probs, axis=1)
  
    ####################################################
    # Step 7: calculate the ks distance between control and test variables 
    # Stack control trends given by simulated counterfactual datasets 
    stack_cntl_culmulate_trend = combined_data_estimate_culmulate_trend[:, :, :, :num_sim_data].mean(axis=3)
  
    ks_test_cntl = np.empty((causal_length, d))
    for dd in range(d):
        for t in range(causal_length):
            ks_test_cntl[t, dd] =  ks_2samp(combined_data_estimate_culmulate_trend[t, dd, :, num_sim_data],
                                            stack_cntl_culmulate_trend[t, dd, :])[0]
  
    print("Done! \n")
    # return result
    return {"mcmc output": mcmc_model_output, "threshold": threshold, "ks test cntl": ks_test_cntl}