
# coding: utf-8

# <h1 align="center"> CS 134: Pset 7 </h1>
# <h4 align="center"> Christine Zhang </h4>

# ---

# In[1]:

import numpy as np
import pandas as pd
import math
import matplotlib as plt
get_ipython().magic(u'matplotlib inline')
import random


# ### 4a

# In[2]:

network_pd = pd.read_csv('network.txt', sep ="\t")
network = network_pd.as_matrix()


# In[3]:

def find_prob (node_a, node_b):
    for i,j,k in network:
        if i == node_a and j == node_b:
            return k
print "Probability of 42 influencing 75: ", find_prob (42, 75)


# ### 4b

# In[4]:

def genRandGraph (weights):
    weights = np.insert(weights, len(weights[0]), 0, axis = 1)
    count = 0
    for i,j,k,l in weights:
        rand = random.uniform(0, 1)
        if rand <= k:
            weights[count][3] = 1
        count += 1
    return weights[weights[:,3] == 1][:,0:3]

rand_graph = genRandGraph(network)


# In[5]:

def avg_num (network):
    trials, total = 100, 0
    for each in range(0, trials):
        rand_graph = genRandGraph(network)
        total += len(set(np.concatenate([rand_graph[:,0], rand_graph[:,1]])))
    return float(total) / float(trials)

print "Average number of nodes: ", avg_num(network)


# ### 4c

# In[345]:

def gen_adj_list (network):
    d = dict()
    for i,j in enumerate(network[:,0]):
        if j in d:
            d[j].append(network[i,1])
        else:
            d[j] = [network[i,1]]
    return d

adj_list = gen_adj_list (rand_graph)


# In[346]:

def bfs(graph, start):
    visited, queue = set(), list(start)
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            if vertex in graph:
                queue.extend([x for x in graph[vertex] if x not in visited])
    return len(list(visited))


# In[347]:

def sampleInfluence(G, S, m):
    nodes = 0
    for each in range(0, m):
        rand_graph = genRandGraph(G)
        adj_list = gen_adj_list (rand_graph)
        nodes += bfs(adj_list, S)
    return nodes / float(m)

print "f(S): ", sampleInfluence(network, [17, 23, 42, 2017], 500)


# ### 4d

# In[ ]:

def greedy (network, m):
    S_nodes, influences = [], []
    for index in range(0, 5):
        influences_inner = []
        for each in set(network[:,0]):
            S_nodes_copy = list(S_nodes)
            S_nodes_copy.append(each)
            f_S = sampleInfluence(network, S_nodes_copy, m)
            influences_inner.append(f_S)
        influences.append(np.amax(influences_inner))
        S_nodes.append(list(set(network[:,0]))[np.argmax(influences_inner)])
    return S_nodes, influences

S_nodes, influences = greedy(network, 10)


# In[382]:

print "f(S): ", influences


# In[381]:

print "Top 5 students: ", S_nodes


# In[ ]:



