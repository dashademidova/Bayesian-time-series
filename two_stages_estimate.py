import numpy as np

def two_stage_estimation(test_data, cntl_index, cntl_data, 
                         graph=False, graph_structure=False, 
                         circle, causal_period, s=0.1,
                         emvs_iteration=50, 
                         v0_value=np.linspace(1e-6, 0.02, 5),
                         mcmc_iterloop=10000, burnin=2000, 
                         stationary=True, 
                         misspecification=False,
                         num_sim_data=30, num_cores=1,
                         seed=1, probs=0.95,
                         plot_EMVS_figure=False,
                         plot_title=None):
  
    T, d = test_data.shape
  
    if T < d:
        test_data = test_data.T
        T, d = test_data.shape

    if graph:
        graph_structure = np.ones((d, d))
    elif not graph_structure:                  
        print("Graph structure must provde!")
        exit()
  
      ########################################################
      ####################### Stage 1 ########################
      ########################################################
      ## Stage 1:
      # EMVS for estimating beta
      # for EMVS, s = 1; for DAEMVS, 0 <= s <= 1
    selection = \
        estimate_counterfactual(test_data=test_data, cntl_index=cntl_index, 
                                cntl_data=cntl_data, graph_structure=graph_structure, 
                                circle=circle, causal_period=causal_period, s=s, 
                                iteration=emvs_iteration, 
                                v0_value=v0_value,
                                stationary=stationary, plot_figure = plot_EMVS_figure,
                                misspecification=misspecification, 
                                plot_title = plot_title)
  
    cntl_term = selection["cntl term"]
    EMVS_estimator = selection["beta hat"]
  
      ########################################################
      ####################### Stage 2 ########################
      ########################################################
      ## Stage 2:
      # MCMC for time varying parameters and covariance and variance matrices
      # Fit into timeseries model
    model_estimates = \
        MultiCausalImpact(test_data=test_data, causal_period=causal_period, 
                          cntl_term=cntl_term, seed=seed, nseasons=circle, 
                          iterloop=iterloop, burnin=burnin, 
                          stationary=stationary, graph=graph, 
                          graph_structure=graph_structure,
                          num_sim_data=num_sim_data, probs=probs,
                          num_cores=num_cores)
  
  # collect result
    mcmc_output = model_estimates["mcmc output"]
    threshold = model_estimates["threshold"]
    ks_test_cntl = model_estimates["ks test cntl"]
  
  
  # return beta_hat,       mcmc_output, ks_test_cntl, threshold
    return {"beta hut": EMVS_estimator, "mcmc output": mcmc_output, 
            "ks test cntl": ks_test_cntl, "threshold": threshold}