import numpy as np

from EMVS import EMVS 

def estimate_counterfactual(test_data, cntl_index, cntl_data, 
                            graph_structure, causal_period, 
                            circle = 7, s = 0.1, iteration = 50,
                            v0_value = np.linspace(1e-6, 0.02, 5),
                            stationary = True, 
                            misspecification = False):
    print('Starting Bayesian EM variable selection...')
    
    nc = test_data.shape[1]*np.int32(cntl_index[0])
    iteration = iteration
    test_data_non_causal = test_data[np.delete(np.arange(test_data.shape[0]), causal_period-1),:]
    cntl_data_non_causal = cntl_data[np.delete(np.arange(cntl_data.shape[0]), causal_period-1),:]

    beta_v0 = np.empty((nc, len(v0_value)))
    v0 = v1 = theta = np.empty(len(v0_value))
    
    if len(v0_value) == 1:
        emvs_result = EMVS(test_data_non_causal, cntl_index, cntl_data_non_causal,
                        graph_structure, circle, v0_value, s, 
                        iteration = iteration, stationary = stationary, 
                        misspecification = misspecification)
        
        beta_v0 = emvs_result['beta'][:, 1]
        theta = emvs_result['theta']
        v1 = emvs_result['v1']
        
    else:
        for i in range(len(v0_value)):
            emvs_result = EMVS(test_data_non_causal, cntl_index, cntl_data_non_causal,
                        graph_structure, circle, v0_value[i], s, 
                        iteration = iteration, stationary = stationary, 
                        misspecification = misspecification)
            
            beta_v0[:,i] = emvs_result['beta'][:, iteration]
            theta[i] = emvs_result['theta'][iteration-1]
            v1[i] = emvs_result['v1']
            
    c = np.sqrt(np.divide(v1, v0_value))
    
    beta_threshold = np.sqrt( (np.log(np.divide(v0_value, v1)) + 
                             2*np.log(np.divide(theta,(1-theta)) + 1e-10)) / (np.divide(1,v1)-np.divide(1,v0_value)))
                           
    
    dCntl = np.sum(cntl_index)
    
    
    
    
    
    
    
    beta_star = beta_threshold[len(v0_value)-1]
    if len(v0_value) > 1:
        beta_hat = beta_v0[:, -1]
    else:
        beta_hat = beta_v0
  
    beta_hat[np.abs(beta_hat) < beta_star] = 0
    
    cntl_term = np.empty((test_data.shape[0], test_data.shape[1]))
    index = 1

    
    for i in range(test_data.shape[1]):
        
        cntl_term[:, i] = cntl_data[:, index-1:index+cntl_index[0]-1] @ beta_hat[index-1:index+cntl_index[0]-1]
        index = index + cntl_index[0]
  
    
    return {'cntl term' : cntl_term, 'beta hat' : beta_hat}

