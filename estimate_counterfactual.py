
# coding: utf-8

# In[ ]:


# the order of input arguments is changed !!!!!!


# In[2]:


import numpy as np

from EMVS import EMVS 


# In[7]:


def estimate_counterfactual(test_data, cntl_index, cntl_data, 
                            graph_structure, causal_period, 
                            circle = 7, s = 0.1, iteration = 50,
                            v0_value = np.linspace(1e-6, 0.02, 5),
                            stationary = True, 
                            misspecification = False,
                            plot_figure = False, plot_title = None):
    print('Starting Bayesian EM variable selection...')
    
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
            
            beta_v0[:,i] = emvs_result['beta'][:, iteration+1]
            theta[i] = emvs_result['theta'][iteration]
            v1[i] = emvs_result['v1']
            
    c = np.sqrt(np.divide(v1, v0_value))
    
    beta_threshold = np.sqrt( (np.log(np.divide(v0_value, v1)) + 
                             2*np.log(np.divide(theta,(1-theta)) + 1e-10)) / (np.divide(1,v1)-np.divide(1,v0.value)))
                           
    
    dCntl = np.sum(cntl_index)
    
    
    
    
    
    
    
    beta_star = beta_threshold[len(v0_value)]
    if len(v0_value) > 1:
        beta_hat = beta_v0[:, -1]
    else:
        beta_hat = beta_v0
  
    beta_hat[np.abs(beta_hat) < beta_star] = 0
    
    cntl_term = np.empty((test_data.shape[0], test_data.shape[1]))
    index = 1

    for i in range(test_data.shape[1]):
        cntl_term[:, i] = cntl_data[:, (index-1):(index+9)] @ beta_hat[(index-1):(index+9)]
        index = index + 10
  
    
    return {'cntl_term' : cntl_term, 'beta_hat' : beta_hat}

