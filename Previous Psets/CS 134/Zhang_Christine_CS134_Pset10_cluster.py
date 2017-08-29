
# coding: utf-8

# <h1 align="center">CS 134: Pset 10 </h1>
# <h4 align="center">Christine Zhang </h4>

# ---

# In[7]:

import numpy as np
import pandas as pd
from random import shuffle
import pickle


# ### Q 4.1

# In[2]:

network_pd = pd.read_csv('data.txt', sep='\t', header= None)
network = network_pd.as_matrix()


# In[4]:

def neighbors (network):
    d = dict()
    for i,j in enumerate(network[:,0]):
        if j in d:
            d[j].append(network[i,1])
        else:
            d[j] = [network[i,1]]
        if network[i,1] in d:
            d[network[i,1]].append(j)
        else:
            d[network[i,1]] = [j]
    return d
adj_lst = neighbors (network)
print "Number of Edges: ", np.shape(network)[0]
print "Number of Nodes: ", len(adj_lst)


# ### Q 4.2

# In[17]:

def floydwarshall(graph):
    n = len(graph)
    dist = np.zeros((n, n))
    for u in graph:
        for v in graph:
            dist[u][v] = 1000
        dist[u][u] = 0
        for neighbor in graph[u]:
            dist[u][neighbor] = 1
    for t in graph:
        print t
        for u in graph:
            for v in graph:
                newdist = dist[u][t] + dist[t][v]
                if newdist < dist[u][v]:
                    dist[u][v] = newdist
    return dist

matrix = floydwarshall(graph)


# In[24]:

counter = []
for i in matrix:
    for j in matrix[i]:
        counter.append(matrix[i][j])
avg_distance = np.sum(counter)/float(len(counter))
print "Average shortest pairs distance: ", avg_distance


# ### Q 4.3

# In[25]:

def dist(p, m, c, norm):
    target = 0
    array_inner = []
    for neighbor in c:
        array_inner.append(m[p][neighbor])
    if norm == "min":
        return np.min(array_inner)
    elif norm == "max":
        return np.max(array_inner)
    else:
        return np.average(array_inner)

print "Min dist: ", dist(5, matrix, [2,8,20], "min")
print "Max dist: ", dist(5, matrix, [2,8,20], "max")
print "Mean dist: ", dist(5, matrix, [2,8,20], "mean")


# ### Q 4.4

# In[31]:

def assign(p, m, c_list, norm):
    dist_lst = []
    for lst in c_list:
        dist_lst.append(int(dist(p, m, lst, norm)))
    return np.argmin(dist_lst)

print "Min dist: ", assign(5, matrix,  [[2, 8, 20], [3, 4, 8, 26]], "min")
print "Mean dist: ", assign(5, matrix,  [[2, 8, 20], [3, 4, 8, 26]], "mean")
print "Max dist: ", assign(5, matrix,  [[2, 8, 20], [3, 4, 8, 26]], "max")


# ### Q 4.5

# In[35]:

def center(m, c):
    distance_lst = []
    for u in c: 
        distance_inner = []
        for v in c:
            distance_inner.append(matrix[u][v]**2)
        distance_lst.append(np.sum(distance_inner))
    return c[np.argmin(distance_lst)], distance_lst[np.argmin(distance_lst)]

center_node, obj_function = center(matrix, [2, 3, 4, 8, 20, 26])
print "Center node: ", center_node


# ### Q 4.6

# In[36]:

def cluster(m, k, norm, i):
    x = adj_lst.keys()
    shuffle(x)
    c_list = {}
    for centroid in x[0:k]:
        c_list[centroid] = [centroid]
        
    for iteration in range(0, i):
        remaining_lst = [j for j in x if j not in c_list]
        for remaining in remaining_lst: 
            c_list.values()[assign(remaining, m, c_list.values(), norm)].append(remaining)
        
        if iteration == (i-1):
            print "Centers: ", c_list.keys()
            for k, cluster in enumerate(c_list.values()):
                center_val, obj_function = center(m, cluster)
                print "Cluster ", c_list.keys()[k], "length: ", len(cluster), "objective function: ", obj_function 
            return c_list

        c_list_new = {}
        for c in c_list.values():
            center_val, obj_function = center(m, c)
            c_list_new[center_val] = [center_val]
        c_list = c_list_new


# In[40]:

for each in [3,5,10,20]:
    for i in range(0, 3):
        print "k-value: ", each, "trial number: ", i+1
        c_list = cluster(matrix, each, "mean", 20)
        c_list = cluster(matrix, each, "min", 20)
        c_list = cluster(matrix, each, "max", 20)


# In[ ]:



