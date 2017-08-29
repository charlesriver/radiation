
# coding: utf-8

# <h1 align="center"> CS 134: Pset 6 </h1>
# <h4 align="center"> Christine Zhang </h4>

# ---

# In[7]:

import numpy as np
import pandas as pd
import math
import matplotlib as plt
get_ipython().magic(u'matplotlib inline')
import pickle
import numpy.linalg as linalg


# ### 5a

# In[8]:

opinions = pickle.load(open('opinions.pk', 'rb'))

print "Number of casacades:", len(opinions)
print "Number of time steps (tau):", len(opinions[0][0])
tau = len(opinions[0][0])
print "Number of nodes:", len(opinions[0].keys())
# tau = 31
# there are 100 cascades
print opinions[6][13].index(1)
# node 13, casacade 6 activated at time step 2


# ### 5b

# In[9]:

network_pd = pd.read_csv('network1.txt', sep=' ')
network = network_pd.as_matrix()

def neighbors (network):
    d = dict()
    for i,j in enumerate(network[:,0]):
        if j in d:
            d[j].append(network[i,1])
        else:
            d[j] = [network[i,1]]
    return d
adj_lst = neighbors(network)
print "Number of nodes: ", len(adj_lst)
adj_lst


# In[10]:

def average_degree (adj_list):
    degrees = [len(adj_lst[each]) for each in adj_lst]
    return np.average(degrees)

print "Average out-degree:", average_degree (adj_lst)


# ### 5c

# In[11]:

def doesInfluence (n, m, t, o):
    if t + 1 < tau:
        if opinions[o][n][t] == 1 and opinions[o][m][t] == 0 and opinions[o][m][t+1] == 1:
            return 1
        else:
            return 0
    else:
        return 0

doesInfluence (10, 4, 1, 0)


# ### 5d

# In[12]:

def MLE (n, m):
    inner_sum, total_sum, count = 0, 0, 0
    for o in range(0, len(opinions)):
        if 1 not in opinions[o][n]:
                    count += 1
        for t in range(0, tau):
            inner_sum += doesInfluence(n, m, t, o) 
            total_sum = len(opinions) - count
    if total_sum != 0:
        return float(inner_sum) / float(total_sum)
    return 0

print MLE (1,2)
print MLE (26,21)


# In[13]:

def MLE_all (network):
    edge_all = []
    for n, m in network:
        edge_all.append(MLE (n, m))
    return list(edge_all)

edge_all = MLE_all (network)


# In[14]:

edge_all = np.resize(edge_all, (len(edge_all), 1))
edge_full = np.concatenate((network, edge_all), axis = 1)
np.savetxt("edge_weights.csv", edge_full, delimiter=",")


# ### 5e

# In[16]:

def average_weight (edge_full):
    edge_avg = []
    for each in set(edge_full[:,0]):
        total = [k for i,j,k in edge_full if i == each]
        edge_avg.append([each, np.average(total)])
    return np.array(edge_avg)
        
edge_avg = average_weight (edge_full)
edge_avg_sorted = edge_avg[edge_avg[:,1].argsort()]

print "Lowest edge weight: ", edge_avg_sorted[0][0]
print "Highest edge weight: ", edge_avg_sorted[-1][0]


# In[59]:

def activation (adj_list):
    activate_all = []
    for n in adj_list:
        activate = []
        for o in range(0, len(opinions)):
            if 1 in opinions[o][n]:
                activate.append(opinions[o][n].index(1))
            else:
                activate.append(tau)
        activate_all.append([n, np.average(activate)])
    return np.array(activate_all)

activate_all = activation(adj_lst)
activate_all_sorted = activate_all[activate_all[:,1].argsort()]

print "Lowest activation: ", activate_all_sorted[0][0]
print "Highest activation: ", activate_all_sorted[-1][0]


# In[3]:

matrix_1 = [[0.1,0.4,0,0.3,0.20], [0,0,0.5,0.5,0], [0.6,0,0.2,0.1,0.1], [0.2,0.2,0.2,0.2,0.2], [0.3,0.3,0.1,0.1,0.2]]

print linalg.matrix_power(matrix_1, 100)
print linalg.matrix_power(matrix_1, 101)


# In[4]:

matrix_2 = [[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.3, 0.4, 0, 0.2],[0.3, 0.1, 0.1, 0.5, 0], [0, 0, 0.5, 0.3, 0.2], [0.2, 0.1, 0.3, 0.2, 0.2]]

print linalg.matrix_power(matrix_2, 100)
print linalg.matrix_power(matrix_2, 101)


# In[5]:

matrix_3 = [[0.1,0.3,0.4,0.2,0], [0,0,0.5,0.4,0.1], [0.2,0.2,0.3,0.2,0.10], [0,0.3,0.5,0.1,0.10],[0.6,0.2,0.1,0,0.1]]

print linalg.matrix_power(matrix_3, 100)
print linalg.matrix_power(matrix_3, 101)


# In[ ]:



