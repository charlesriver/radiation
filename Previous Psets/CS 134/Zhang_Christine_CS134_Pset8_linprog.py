
# coding: utf-8

# <h1 align="center"> CS 134: Pset 8 </h1>
# <h4 align="center"> Christine Zhang </h4>

# ---

# In[1]:

import numpy as np
import pandas as pd
import math
import matplotlib as plt
get_ipython().magic(u'matplotlib inline')
import pickle
import scipy.optimize as sc


# ### 5a

# In[25]:

opinions = pickle.load(open('pset8_opinions.pkl', 'rb'))

num_cas = len(opinions)
print "Number of diffusion processes:", num_cas
print "Number of nodes: ", len(opinions[0])
tau = len(opinions[0][0])
print "Number of time steps (tau):", tau


# ### 5b

# In[226]:

network_pd = pd.read_csv('pset8_network1.txt', sep=' ', header= None)
network = network_pd.as_matrix()

def neighbors (network):
    d = dict()
    
    # read in network
    for i,j in enumerate(network[:,0]):
        if j in d:
            d[j].append(network[i,1])
        else:
            d[j] = [network[i,1]]
            
    # append self loops
    for i in range(0, len(opinions)):
        if i in d:
            d[i].append(i)
            d[i] = sorted(d[i])
        else:
            d[i] = [i]
    return d


adj_list = neighbors(network)
adj_list 


# In[192]:

def average_degree (adj_list):
    degrees = [len(adj_lst[each]) for each in adj_lst]
    return np.average(degrees)

print "Average out-degree:", average_degree (adj_lst)


# ### 5d

# In[238]:

def genmatrices (node_u):
    LHS, RHS = [], []
    neighbors = adj_list[node_u]
    for cascade_index in range(0, num_cas):
        for t in range(1, tau):
            LHS_inner = list()

            if opinions[cascade_index][node_u][t] == 1 and opinions[cascade_index][node_u][t-1] == 0:
                for i in range(0, len(opinions[cascade_index][node_u])):
                    if i in neighbors:
                        if opinions[cascade_index][i][t-1] == 1:
                            LHS_inner.append(-1)
                        else:
                            LHS_inner.append(0)
                RHS.append(-0.5)

            if opinions[cascade_index][node_u][t] == 0 and opinions[cascade_index][node_u][t-1] == 1:
                for i in range(0, len(opinions[cascade_index][node_u])):
                    if i in neighbors:
                        if opinions[cascade_index][i][t-1] == 1:
                            LHS_inner.append(1)
                        else:
                            LHS_inner.append(0)
                RHS.append(0.5)
            
            if LHS_inner != []:
                LHS.append(LHS_inner)
    
    if len(RHS) == 0:
        return 0
    
    C = [-1]*np.shape(LHS)[1]
    weights_LHS = [[1]*np.shape(LHS)[1]]
    weights_RHS = [1]
    
    res = sc.linprog(C, A_ub=LHS, b_ub=RHS, A_eq = weights_LHS, b_eq = weights_RHS, bounds = (0,1), method='simplex')
    return res


# In[284]:

influences = []
for node_u in adj_list:
    res = genmatrices(node_u)
    if res != 0: 
        for i, neigh in enumerate(adj_list[node_u]):
            influences.append([node_u, neigh, res.x[i]])
    else:
        for neigh in adj_list[node_u]:
            influences.append([node_u, neigh, 0])
influences = np.array(influences)
            
np.savetxt('influences.csv', influences, delimiter = ",")


# ### 5e

# In[278]:

def average_influence (influences):
    influence_avg = []
    for each in set(np.array(influences)[:,0]):
        total = [k for i,j,k in influences if j == each]
        influence_avg.append([each, np.average(total)])
    return np.array(influence_avg)
        
influence_avg = average_influence (influences)
influence_avg_sorted = influence_avg[influence_avg[:,1].argsort()]

print "Top 5 students: ", influence_avg_sorted[-5:][:,0][::-1]


# In[ ]:



