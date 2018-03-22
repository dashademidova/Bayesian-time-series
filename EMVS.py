import random
import numpy as np
import pandas as pd
import scipy as sp

import rpy2.robjects as R
from rpy2.robjects.packages import importr

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


from scipy.stats import wishart
from scipy.linalg import block_diag

from statsmodels.tsa.api import VAR

BDgraph = importr('BDgraph')
base = importr('base')
stats = importr('stats')

from koopmanfilter import koopmanfilter
from kalmflter import kalmflter
from MungeMatrix import MungeMatrix

def EMVS(test, cntl_index, cntl, graph_structure, circle,
                 v0 = 0.1, s = 0.1, iteration = 50, stationary = True, 
                 misspecification = False):
    ##################### EMVS ########################
    # organize dataset
    length = test.shape[0]
    n = test.shape[1]  
    dCntl = sum(cntl_index)
    cntl_input = cntl.copy()

    # re-organize cntl dataset
    cntl = None
    index = 0

    cntl = np.empty((cntl_input.shape[0], cntl_index.max(),n))


    for dim in range(n):
        cntl[:,:,dim] = cntl_input[:, index:index+cntl_index[dim]]
        index = index+cntl_index[dim]

    ############################################################
    # rename y.est to y.matrix and cntl.est to x.matrix

    x_std = np.zeros((length,cntl_index[0],n))


    for i in range(test.shape[1]):
        x_std[:,:,i] = cntl[:,:,i] - np.tile(np.mean(cntl[:,:,i], axis=0),(cntl.shape[0],1))

    x_matrix = np.zeros((n*length, dCntl))
    index = 0

    for dims in range(n) :
        x_matrix[np.arange(dims,n*length,n), 
                 index:(index+cntl_index[dims])] = x_std[:,:, dims]
        index = index + cntl_index[dims]

    y_std = test - np.tile(np.mean(test, axis=0),(test.shape[0],1))
    y_matrix = y_std.T.reshape(y_std.shape[0]*y_std.shape[1], order='F')

    # giving starting values for beta and sigma
    beta_hat = np.zeros(dCntl)
    y_tilde = (y_matrix - x_matrix @ beta_hat).reshape((n,length),order='F')

    sigma_hat = (y_std.T @ y_std) / (length-1)

    if np.linalg.matrix_rank(sigma_hat) != sigma_hat.shape[0]:
        sigma_hat = MungeMatrix(sigma_hat)

    sigma_hat_inv = np.linalg.solve(sigma_hat, np.eye(sigma_hat.shape[0]))
    sigma_hat_inv[graph_structure == 0] = 0
    B = np.eye(n) # prior parameters for wishart priors
    delta = n + 1 # prior parameter for wishart priors


    # giving staring points for v0, v1, a, b, theta
    v0 = v0 # initial v0
    # v1 <- 100 # initial v1
    v1 = 10
    a = 1 # intial a, shape1 in beta distribution
    b = 1  # intial b, shape2 in beta distribution
    theta = np.random.beta(a,b) # intial theta from beta distribtuion


    if misspecification == False :
        # giving starting values for Q
        k1 = k2 = k3 = 0.1 # prior parameters for sigma.u, sigma.v, sigma.w
        sigma_u_hat = np.array(base.chol2inv(np.array(BDgraph.rgwish(adj_g = graph_structure, 
                                                   b = n+1, D = (k1**2) * n * np.eye(n)))[:,:,0]))

        sigma_v_inv = np.array(BDgraph.rgwish(adj_g = graph_structure, 
                                                   b = n+1, D = (k2**2) * n * np.eye(n)))[:,:,0]
        sigma_v_hat = np.array(base.chol2inv(sigma_v_inv))
        sigma_v_hat_inv = np.linalg.solve(sigma_v_hat, np.eye(n))
        sigma_w_hat = np.array(base.chol2inv(np.array(BDgraph.rgwish(adj_g = graph_structure, 
                                                   b = n+1, D = (k3**2) * n * np.eye(n)))[:,:,0]))
        Q_hat =  block_diag(sigma_u_hat, sigma_v_hat, sigma_w_hat)
        Q_inv = np.linalg.solve(Q_hat, np.eye(Q_hat.shape[0]))



        # giving initial parameters for KFBS (Kalman filter and backward simulation)
        # initial a.int
        a_int = np.zeros(n*(circle+1)) 
        # initialize P.int
        P_int = np.zeros((n*(circle+1), n*(circle+1)))

        if (stationary == True) :
            P_int[:(3*n),:(3*n)] = np.eye(3*n)
        else:
            P_int[:(3*n), :(3*n)] = np.eye(3*n)*(10**6)


        # initial transition matrix
        trans = np.zeros(((circle+1)*n, (circle+1)*n))
        linear = np.eye(2*n)
        linear[:n, n:(2*n)] = np.eye(n)
        trans[:(2*n), :(2*n)] = linear

        # take initial variance of tau from the data
        if stationary == True:
            #data_yw = stats.ar_yw(test, aic = False, order_max = 1,
            #                 demean = T, intercept = T) #????????????????????????????????????????????????
            #phi_hat = np.array(data_yw[1]).reshape((n,n))
            #phi_hat = data_yw_ar #????????????????????????????????????????????????
            var = VAR(test)
            phi_hat = var.fit(maxlags=1, ic=None, trend="nc").params
        else:
            phi_hat = np.eye(n)

        trans[n:(2*n), n:(2*n)] = phi_hat
        seasonal = np.zeros((circle-1, circle-1))
        seasonal[0, :] = -1
        seasonal[1:(circle-1), 0:(circle-2)] = np.eye(circle-2)


        for dims in range(n):
            trans[(2*n+dims):(circle+1)*n:n,(2*n+dims):(circle+1)*n:n] = seasonal

        # define
        z = np.zeros((n*(circle+1), n))
        z[:n, :] = np.eye(n)
        z[(2*n):(3*n), :] = np.eye(n)

        # initialize R 
        R = np.zeros((n*(circle+1), n*3))
        R[:(3*n), :(3*n)] = np.eye(3*n)

        # initialize alpha.hat
        a_hat = np.zeros((n*(circle+1), length))

    # step 2: EM update parameters
    # create matrix to collect results
    iterloop = iteration
    beta_update = np.zeros((dCntl, iterloop+1))
    sigma_update = np.empty( (n,n,iterloop)) 
    v0_update = v1_update = theta_update = np.empty(iterloop)
    lp_update = np.empty(iterloop)

    if misspecification == False:
        a_update = np.empty((length, n*(circle+1),iterloop))
        phi_update = np.empty((n,n,iterloop))
        sigma_u_update = sigma_v_update = sigma_w_update = np.empty((n,n,iterloop)) 


    for iter in range(iterloop):
        # ----------------- E-step ------------------- #

        if misspecification == False : 
            # upsing kalman filter and backward smoother
            ## update alpha
            KFBS = koopmanfilter(n*(circle+1), y_tilde.T, trans, z, a_int, P_int, 
                                sigma_hat, Q_hat, R, output_var_cov = True)
            a_hat = KFBS['a sample'] 
            P_hat = KFBS['P sample'] 
            P_cov_hat = KFBS['P cov sample'] 


            # calculate expectation for E(alpha_t alpha_t') and E(alpha_t alpha_(t-1)')
            V_hat = np.zeros((n*(circle+1),n*(circle+1),length))
            V_cov_hat = np.zeros((n*(circle+1),n*(circle+1),length-1))
            for i in range(length):
                V_hat[:,:,i] = P_hat[:,:,i] + np.outer(a_hat[:,i],a_hat[:,i])
                if i < length-1:
                    V_cov_hat[:,:,i] = P_cov_hat[:,:,i] + np.outer(a_hat[:,i], a_hat[:,i+1])

        # ----------------- M-step ------------------- #
        ## update A
        gamma1 = np.array(stats.dnorm(beta_hat, mean = 0, sd = np.sqrt(v1)))
        gamma2 = np.array(stats.dnorm(beta_hat, mean = 0, sd = np.sqrt(v0)))
        pstar = np.divide(np.power((theta*gamma1),s), (np.power((theta*gamma1),s) + np.power((gamma2*(1-theta)),s)))

        A = np.eye(dCntl)
        np.fill_diagonal(A, (1-pstar)/v0 + pstar/v1)


        ## update beta
        # organize dataset
        kron_sigma_inverse = np.kron(np.eye(length), sigma_hat_inv) 
        XcovX = (x_matrix.T @ kron_sigma_inverse) @ x_matrix  

        if misspecification == False: 
            a_z = ((a_hat.T @ z).T).reshape((-1,1),order='F')[:,0]
            XcovY = (x_matrix.T @ kron_sigma_inverse) @ (y_matrix - a_z)

        else: 
            XcovY = (x_matrix.T @ kron_sigma_inverse) @ (y_matrix) 


        beta_hat = (np.linalg.solve(XcovX + A, XcovY))#.reshape((-1,1),order='F')

        gamma1 = np.array(stats.dnorm(beta_hat, mean = 0, sd = np.sqrt(v1)))
        gamma2 = np.array(stats.dnorm(beta_hat, mean = 0, sd = np.sqrt(v0)))
        pstar = np.divide(np.power((theta*gamma1),s), (np.power((theta*gamma1),s) + np.power((gamma2*(1-theta)),s)))

        # update theta
        theta = (sum(pstar) + a - 1) / (a + b + dCntl - 2)

        if misspecification == False: 
            # update phi
            # vec(phi) ~ N(0, 0.01*I_{n^2}) prior of vec(phi)
            if stationary == True: 
                phi_term1 = phi_term2 = 0
                for i in range(length-1): 

                    phi_term1 = phi_term1 + np.kron((V_hat[n:(2*n), n:(2*n), i]).T, sigma_v_hat_inv)
                    phi_term2 = phi_term2 + np.kron((V_cov_hat[n:(2*n),n:(2*n), i]).T, sigma_v_hat_inv) 



                vec_phi = np.linalg.solve(phi_term1 + 10*np.eye(n**2), phi_term2 @ np.eye(n).reshape((-1,1), order='F')) #?????????????????????????????????
                phi_hat = vec_phi.reshape((n,n), order='F')

            else:
                phi_hat = np.eye(n)


            # update trans
            trans[n:(2*n), n:(2*n)] = phi_hat

            # update sigma
            y_tilde =  (y_matrix - (x_matrix @ beta_hat)).reshape((n, length), order='F') 
            P_plus_aa = 0
            for i in range(length): 
                P_plus_aa = P_plus_aa + V_hat[:,:,i] 


            sigma_mat = y_tilde @ y_tilde.T - z.T @ a_hat @ y_tilde.T - y_tilde @ a_hat.T @ z + z.T @ P_plus_aa @ z
            sigma_hat = (sigma_mat + B) / (length + delta - 2) 
            sigma_hat_inv = np.linalg.solve(sigma_hat, np.eye(sigma_hat.shape[0]))


            #sigma.hat.inv[graph.structure == 0] <- 0 
            if min(abs(np.linalg.eigvals(sigma_hat_inv))) <= 0:
                sigma_hat_inv = MungeMatrix(sigma_hat_inv)

            sigma_hat = np.linalg.solve(sigma_hat_inv, np.eye(sigma_hat_inv.shape[0])) 


            # update Q
            Q_mat_term = 0
            for i in range(length-1): 
                Q_mat_term = Q_mat_term + V_hat[:,:,i+1] - trans @ V_cov_hat[:,:,i] - V_cov_hat[:,:,i].T @ trans.T + trans @ V_hat[:,:,i] @ trans.T

            Q_mat = R.T @ Q_mat_term @ R
            Q_hat = (Q_mat + block_diag((n+1)*k1**2*B, (n+1)*k2**2*B, (n+1)*k3**2*B)) / (length + delta - 3)

            # update sigma.u
            sigma_u_hat = Q_hat[:n, :n]
            sigma_u_hat_inv = np.linalg.solve(sigma_u_hat, np.eye(sigma_u_hat.shape[0]))

            # sigma.u.hat.inv[graph.structure == 0] <- 0
            if min(abs(np.linalg.eigvals(sigma_u_hat_inv))) <= 0:
                sigma_u_hat_inv = MungeMatrix(sigma_u_hat_inv)


            # update sigma.v
            sigma_v_hat = Q_hat[n:(2*n), n:(2*n)]
            sigma_v_hat_inv = np.linalg.solve(sigma_v_hat, np.eye(sigma_v_hat.shape[0]))
            # sigma.v.hat.inv[graph.structure == 0] <- 0
            if min(abs(np.linalg.eigvals(sigma_v_hat_inv))) <= 0:
                sigma_v_hat_inv = MungeMatrix(sigma_v_hat_inv)

            # update sigma.w
            sigma_w_hat_inv = np.linalg.solve(Q_hat[(2*n):(3*n), (2*n):(3*n)], np.eye(n))
            # sigma.w.hat.inv[graph.structure == 0] <- 0
            if min(abs(np.linalg.eigvals(sigma_w_hat_inv))) <= 0:
                sigma_w_hat_inv = MungeMatrix(sigma_w_hat_inv)


            # update Q.hat
            Q_inv = block_diag(sigma_u_hat_inv, sigma_v_hat_inv, sigma_w_hat_inv)
            Q_hat = np.linalg.solve(Q_inv, np.eye(Q_inv.shape[0]))


        else: 

            # ------------- FOR MISSPECIFIED MODEL ---------------- #
            # update sigma
            y_tilde = np.full((n, length), y_matrix - x_matrix @ beta_hat)
            sigma_mat = y_tilde @ y_tilde.T
            sigma_hat = (sigma_mat + B) / (length + delta - 2) 
            sigma_hat_inv = np.linalg.solve(sigma_hat, np.eye(sigma_hat.shape[0]))
            #sigma.hat.inv[graph.structure == 0] = 0
            if min(abs(np.linalg.eigvals(sigma_hat_inv))) <= 0:
                sigma_hat_inv = MungeMatrix(sigma_hat_inv)
            sigma_hat = np.linalg.solve(sigma_hat_inv, np.eye(sigma_hat_inv.shape[0]))


        # collect result
        if misspecification == False:
            a_update[:,:, iter] = a_hat.reshape((-1,1),order='F').reshape((a_hat.shape[1], a_hat.shape[0]), order='F')
            phi_update[:,:,iter] = phi_hat

        beta_update[:, iter+1] = beta_hat
        sigma_update[:,: , iter] = sigma_hat
        theta_update[iter] = theta
        
    #return(list(beta = beta.update, theta = theta.update, v1 = v1))
    return {'beta': beta_update, 'theta': theta_update, 'v1':v1}

