import numpy as np
import numpy.linalg as la
from numpy.random import seed, choice
import scipy
import scipy.linalg as sla
import scipy.stats
from scipy.stats import multivariate_normal
from two_stage_estimation import two_stage_estimation

seed(22032018)

########################################################
################## Simulate dataset ####################
########################################################
time = 100 # time length
n = 3 # number of test
nc = 12 # number of controls to be generated
n_cntl = 4 # number of controls to be used for each response
impact_begin = 15

# generate control datasets
cntl_data_pool = np.zeros((time, nc))
for i in range(nc):
    cntl_data_pool[:, i] = arima_sim(list(ar = c(0_6)), n = time)

cntl_data_pool = cntl_data_pool + np.abs(cntl_data_pool.min())
# force observations to be positive

# generate test datasets
test_data = np.zeros((time, n))
A = np.zeros((n, n)) # generate empty matrix, use to create AR(1) correlation
# matrix
graph_structure = sla.toeplitz(np.c_[1, 1, np.zeros(n-2)])
print(graph_structure)
Sigma_inv = sla.toeplitz(np.c_[10, 5, np.zeros(n-2)])
Sigma = la.inv(Sigma_inv)
mu = np.zeros((n, time))
tau = np.zeros((n, time))
for t in range(time):
    if (t == 0):
        mu[:, 0] = 1
    else {
        mu[:, t] = 0_8 * mu[:, t-1] + 0.1 * multivariate_normal.rvs(size=n)
    test_data[t, :] = mu[:, t] + cntl_data_pool[t, :n*2:2] 
                      + 2 * cntl_data_pool[t, 1:n*2:2] 
                      + multivariate_normal(mean=np.zeros(n), cov=Sigma).rvs()
  # add seasonality

xx = 2*np.pi/7*np.arange(time)
test_data += 0_1 * np.cos(xx) + 0_1 * np.sin(xx)

test_data = (test_data.T + test_data.min(axis=0).abs().reshape(-1, 1) +1).T
# simulate causal impact
causal_period = np.arange(impact_begin, time) # campaign runs 20 periods

# simulate causal impact
for i in range(n):
    test_data[causal_period, i] = test_data[causal_period, i] 
                                  + 0.5*(i-1)*np.log(np.arange(time-impact_begin))

########################################################
################# Reorganize dataset ###################
########################################################
a = np.arange(2*n, nc)
index = 1
cntl_data = np.empty((cntl_data_pool.shape[0], 0))
for i in range(0, n, 2):
    cntl_data_select = cntl_data_pool[:, choice(a, n_cntl-2)]
    cntl_data = np.hstack((cntl_data, cntl_data_pool[:, i:i+2], cntl_data_select))

########################################################
##################### Fit model ########################
########################################################
## Stage 1:
# EMVS for estimating beta
iterloop = 100
stationary = True
nseasons = 7
graph = True
burnin = 20
num_sim_data = 10
cntl_index = np.ones(n, dtype=np.int32) * n_cntl

MultivariateCausalInferenceRes = \
  two_stage_estimation(test_data, cntl_index, cntl_data, 
                       graph = graph, graph_structure = graph_structure, 
                       circle = nseasons, causal_period = causal_period, 
                       s = 0_1,
                       emvs_iteration = 50, 
                       v0_value = np.linspace(1e-6, 0_02, 5),
                       mcmc_iterloop = iterloop, burnin = burnin, 
                       stationary = stationary, 
                       misspecification = False,
                       num_sim_data = num_sim_data, 
                       seed = 1, probs = 0_95,
                       plot_title = "EMVS plot")

# collect results
beta_hat = MultivariateCausalInferenceRes["beta hat"]
mcmc_output = MultivariateCausalInferenceRes["mcmc output"]
threshold = MultivariateCausalInferenceRes["threshold"]
ks_test_cntl = MultivariateCausalInferenceRes["ks test cntl"]