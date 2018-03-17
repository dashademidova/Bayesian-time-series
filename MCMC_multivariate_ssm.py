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
import numpy as np
import scipy
import scipy.stats
from scipy.stats import wishart
import scipy.linalg as sla

def MCMC.multivariate.ssm(test_data, causal_period, nseasons=12, iterloop = 1000, 
                          burnin=100, stationary=True, graph=False, graph_structureNone):

    
    ############## load functions ###############
    if stationary:
        import stationaryRestrict
    
    ############### organize dataset #################
    length, d = test_data.shape
    # seperate causal period dataset
    causal_period = causal_period
    causal_period_len = causal_period.shape[0]
    length_non_causal = length - causal_period_len
    
    ############### initialize parameters #################
    # initialize z
    circle = np.min(nseasons, length)
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
    sigma_hat_inv = wishart(adj.g = graph.structure, b = d+1, 
                            D = 0.1^2 * d * diag(d))[,,1]

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
        data_yw = ar.yw(as.matrix(test.data[causal.period, ]), 
                       aic = FALSE, order.max = 1,
                       demean = T, intercept = T)
        data_phi = np.hstack(data.yw$ar).reshape(d, d)
        trans[d:2*d, d:2*d] = data_phi
    else:
        trans[d:2*d, d:2*d] = np.eye(d)

    seasonal = np.zeros((circle-1, circle-1))
    seasonal[0, :] = -1
    seasonal[1:circle-1, :circle-2] = np.eye(circle-2)
    for dims in range(d):
        trans[np.arange(2*d+dims, n, d), np.arange(2*d+dims, n, d)] = seasonal
    
    # initialize R 
    R = np.zeros((n, d*3))
    R[:3*d, :3*d] = np.eye(3*d)
    
    # initialize covariance matrix Q = bdiag(sigmaU, sigmaV, sigmaW)
    k1, k2, k3 = 0.1, 0.1, 0.1
    # what's happening here
    if not graph:
      sigmaU <- chol2inv(rWishart(1, d+1, k1^2 * d * diag(d))[,,1])
      sigmaV.inv <- rWishart(1, d+1, k2^2 * d * diag(d))[,,1]
      sigmaV <- chol2inv(sigmaV.inv)
      sigmaW <- chol2inv(rWishart(1, d+1, k3^2 * d * diag(d))[,,1])
    } else {
      sigmaU <- chol2inv(rgwish(adj.g = graph.structure,
                                b = d+1, D = k1^2 * d * diag(d))[,,1])
      sigmaV.inv <- rgwish(adj.g = graph.structure,
                           b = d+1, D = k2^2 * d * diag(d))[,,1]
      sigmaV <- chol2inv(sigmaV.inv)
      sigmaW <- chol2inv(rgwish(adj.g = graph.structure,
                                b = d+1, D = k3^2 * d * diag(d))[,,1])
    }
    Q = sla.block_diag(sigmaU, sigmaV, sigmaW)
    
    ################### Prepare for MCMC Sampling ##################
    # mcmc iteration and burnin
    iterloop = iterloop 
    burnin = burnin
    
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
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! I stop here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # pb  <- txtProgressBar(1, iterloop, style=3)    # report progress
    print("\nStarting MCMC sampling: \n")     # report progress
    ##################### Begin MCMC Sampling #######################
    # ptm <- proc.time()
    for itery in range(iterloop):
      
      # report progress
      # setTxtProgressBar(pb, iter)
      
      ## --------------------------------------- ##
      ## Step 1. obtain draws of alpha, apply Koopman's filter (2002)
      # simulate w.hat, y.hat, alpha.hat for Koopman's filter (2002)
      alpha_plus <- Matrix(0, length, n)
      for (t in 1:length) {
        eta <- mvrnorm(1, mu = rep(0, 3*d), Q)
        if (t == 1) {
          alpha.plus[t, ] <- mu.ss + trans %*% alpha.int + R %*% eta
        }
        else {
          alpha.plus[t, ] <- mu.ss + trans %*% alpha.plus[t-1, ] + R %*% eta
        }
      }
      test.est.plus <- alpha.plus %*% z + 
        mvrnorm(n = length, mu = rep(0, d), Sigma = sigma.hat)
      test.est.star <- test.data - test.est.plus 
      # Estimate alpha parameters
      sample.alpha.draws <- 
        koopmanfilter(n, test.est.star, trans, z, aStar.int, 2*P.int, 
                      2*sigma.hat, 2*Q, R, causal.period)
      alpha.star.hat <- sample.alpha.draws$alpha.sample
      alpha.draws <- alpha.star.hat + alpha.plus
      
      # collect a.last and P.last, 
      # use them for starting point of koopman filter for causal period dataset
      a.last.sample[, iter] <- as.vector(sample.alpha.draws$a.last)
      P.last.sample[, , iter] <- as.matrix(sample.alpha.draws$P.last)
      
      ## ---------------------------------------- ##
      ## Step 2: make stationary restriction
      if (stationary == TRUE) {
        alpha.draws.tau <- alpha.draws[1:length.non.causal, (d+1):(d*2)]
        if (iter == 1){
          alpha.draws.tau.demean <- alpha.draws.tau
          Theta.draw <- stationaryRestrict(as.matrix(alpha.draws.tau.demean),
                                           sigmaV, sigmaV.inv)
        } else{
          alpha.draws.tau.demean <- t(t(alpha.draws.tau) - D.draw)
          Theta.draw <- stationaryRestrict(as.matrix(alpha.draws.tau.demean),
                                           sigmaV.draws, sigmaV.inv)
        }
        trans[(d+1):(2*d), (d+1):(2*d)] <- Theta.draw
        
        ## ---------------------------------------- ##
        ## Step 3: sample intercept mu.D, denote N(0, I) prior for D
        tau.part.A <- alpha.draws.tau[2:length.non.causal, ] -
          alpha.draws.tau[1:(length.non.causal-1), ] %*% t(Theta.draw)
        tau.part.B <- diag(d) - Theta.draw
        D.var <- solve(
          (length.non.causal-1)*crossprod(tau.part.B, sigmaV.inv) %*% tau.part.B + 
            diag(d))
        D.mean <- D.var %*% (crossprod(tau.part.B, sigmaV.inv) %*% 
                               colSums(tau.part.A))
        D.draw <- mvrnorm(mu = D.mean, Sigma = D.var)
        # update the mean: D - theta * D
        D.mu <- tau.part.B %*% D.draw
        mu.ss[(d+1):(2*d)] <- D.mu
        # update alpha.draws.tau.demean
        alpha.draws.tau.demean <- t(t(alpha.draws.tau) - D.draw)
      }
      
      ## ---------------------------------------- ##
      ## Step 4: update sigmaU, sigmaV, sigmaW
      # parameter in sigmaU
      PhiU <- crossprod(alpha.draws[2:length.non.causal, 1:d] - 
                          alpha.draws[1:(length.non.causal-1), 1:d] - 
                          alpha.draws[1:(length.non.causal-1), (d+1):(d*2)])
      PhiU <- matrix(PhiU, d, d)
      # parameter in sigmaV
      if (stationary == TRUE) {
        PhiV <- crossprod(alpha.draws.tau.demean[2:length.non.causal, ] -
                            alpha.draws.tau.demean[1:(length.non.causal-1), ] %*%
                            t(Theta.draw))
      } else {
        PhiV <- crossprod(alpha.draws[2:length.non.causal, (d+1):(2*d)] -
                            alpha.draws[1:(length.non.causal-1), (d+1):(2*d)])
      }
      PhiV <- matrix(PhiV, d, d)
      # parameter in sigmaW
      bind.W <- NULL
      for (dims in 1:d) {
        bind.W <- cbind(bind.W, rowSums(
          cbind(alpha.draws[2:length.non.causal, seq(d*2+dims, n, by=d)],
                alpha.draws[1:(length.non.causal-1), n-d+dims])))
      }
      PhiW <- crossprod(bind.W)
      PhiW <- matrix(PhiW, d, d)
      scale.U <- PhiU + (d+1)*k1^2*diag(d)
      scale.V <- PhiV + (d+1)*k2^2*diag(d)
      scale.W <- PhiW + (d+1)*k3^2*diag(d)
      # sample sigmaU, sigmaV, sigma W from their posteriors
      if (graph == FALSE) {
        sigmaU.draws <- solve(rWishart(1, length.non.causal+d-1, scale.U)[,,1])
        sigmaV.inv <- rWishart(1, length.non.causal+d-1, scale.V)[,,1]
        sigmaV.draws <- solve(sigmaV.inv)
        sigmaW.draws <- solve(rWishart(1, length.non.causal+d-1, scale.W)[,,1])
      } else {
        sigmaU.draws <- solve(rgwish(adj.g = graph.structure, 
                                     b = length.non.causal+d-1, D = scale.U)[,,1])
        sigmaV.inv <- rgwish(adj.g = graph.structure, 
                             b = length.non.causal+d-1, D = scale.V)[,,1]
        sigmaV.draws <- solve(sigmaV.inv)
        sigmaW.draws <- solve(rgwish(adj.g = graph.structure, 
                                     b = length.non.causal+d-1, D = scale.W)[,,1])
      }
      Q <- bdiag(sigmaU.draws, sigmaV.draws, sigmaW.draws)
      
      ## ---------------------------------------- ##
      ## Step 5: update sigma.hat
      res <- (test.data - alpha.draws %*% z)[1:length.non.causal, ]
      if (graph == FALSE) {
        D.sigma <- matrix(crossprod(res) + B, d, d)
        sigma.hat.inv <- rWishart(1, delta+length.non.causal, D.sigma)[,,1]
        sigma.hat <- solve(sigma.hat.inv)
      } else {
        D.sigma <- matrix(crossprod(res) + B, d, d)
        sigma.hat.inv <- rgwish(n=1, adj.g = graph.structure, 
                                b = (delta+length.non.causal), D = D.sigma)[,,1]
        sigma.hat <- solve(sigma.hat.inv)
      }
      
      ## ---------------------------------------- ##
      ## Step 6: estimating dataset using predicted value
      prediction.sample[, , iter] <- as.matrix(alpha.draws %*% z) + 
        mvrnorm(T, mu = rep(0, d), sigma.hat)
      ## ---------------------------------------- ##
      ## Step 7: collect sample draws
      mu.sample[, , iter] <- matrix(alpha.draws, length, d) 
      if (stationary == T) {
        Theta.sample[, , iter] <- Theta.draw
        D.sample[, iter] <- D.draw
      }
      sigma.sample[, , iter] <- sigma.hat
      sigma.U.sample[, , iter] <- sigmaU.draws
      sigma.V.sample[, , iter] <- sigmaV.draws
      sigma.W.sample[, , iter] <- sigmaW.draws
    }
    # return result
    if (stationary == T) {
      list(prediction.sample = prediction.sample, mu.sample = mu.sample,
           Theta.sample = Theta.sample, D.sample = D.sample, 
           sigma.sample = sigma.sample, sigma.U.sample = sigma.U.sample,
           sigma.V.sample = sigma.V.sample, sigma.W.sample = sigma.W.sample,
           a.last.sample = a.last.sample, P.last.sample = P.last.sample,
           z = z, R = R, trans = trans)
    } else {
      list(prediction.sample = prediction.sample, mu.sample = mu.sample,
           sigma.sample = sigma.sample, sigma.U.sample = sigma.U.sample,
           sigma.V.sample = sigma.V.sample, sigma.W.sample = sigma.W.sample,
           a.last.sample = a.last.sample, P.last.sample = P.last.sample,
           z = z, R = R, trans = trans)
    }
}