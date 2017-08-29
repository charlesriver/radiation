
# coding: utf-8

# <h1 align="center"> CS134: Networks Pset 9 </h1>
# <h4 align="center"> Christine Zhang </h4>

# ---

# In[3]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# ### Q2

# In[109]:

google_pd = pd.read_csv('google.txt', sep=' ', header= None)
google = google_pd.as_matrix()


# In[50]:

def neighbors (network):
    d = dict()
    for i,j in enumerate(network[:,0]):
        
        if j in d:
            d[j].append(network[i,1])
        else:
            d[j] = [network[i,1]]
        if network[i,1] not in d:
            d[network[i,1]] = []
            
    return d


adj_list = neighbors(google)
print "Number of nodes: ", len(adj_list) 


# In[51]:

def average_degree (adj_list):
    degrees = [len(adj_list[each]) for each in adj_list]
    return np.average(degrees)

print "Average out-degree:", average_degree (adj_list)


# ### Q3

# In[82]:

def create_d (g):
    d = dict()
    totalnodes = float(len(g))
    for i in g:
        d[i] = 1/totalnodes
    return d
d = create_d (adj_list)


# In[83]:

def pageRankIter(g,d):
    d_new = dict()
    for i in g:
        if i in d:
            degree = float(len(g[i]))
            if degree == 0.0:
                d_new[i] = d[i]
            else:
                for j in g[i]:
                    if j in d_new:
                        d_new[j] += d[i]/degree
                    else:
                        d_new[j] = d[i]/degree  
        else:
            d_new[i] = 0
    return d_new
d_new = pageRankIter(adj_list, d)


# In[105]:

def plothist (d, title):
    plt.hist(d.values(), 50, range = (0, 0.0001))
    plt.xlabel('PR Score')
    plt.ylabel('Number of Web Pages')
    plt.title('PR Score Distribution with k = ' + str(title))
    plt.show()
plothist(d_new, 1)


# ### Q4

# In[106]:

def basicPR(g,d,k):
    for i in range(0, k):
        d_rec = pageRankIter(g, d)
        d = d_rec
    return d
d_10 = basicPR(adj_list, d, 10)
d_50 = basicPR(adj_list, d, 50)
d_200 = basicPR(adj_list, d, 200)


# In[107]:

plothist(d_10, 10)
plothist(d_50, 50)
plothist(d_200, 200)


# ### Q5

# In[101]:

def scaledPR(g,d,k,s):
    for i in range(0, k):
        d_new = dict()
        n = len(d)
        for i in g:
            for j in g[i]:
                if i in d:
                    if j in d_new:
                        d_new[j] += 1/float(len(g[i]))*d[i]*s
                    else:
                        d_new[j] = 1/float(len(g[i]))*d[i]*s 
        for i in d_new:
            d_new[i] += (1-s)/float(n)
        d = d_new
    return d

d_10_s = scaledPR(adj_list, d, 10, 0.85)
d_50_s = scaledPR(adj_list, d, 50, 0.85)
d_200_s = scaledPR(adj_list, d, 200, 0.85)


# In[108]:

plothist(d_10_s, 10)
plothist(d_50_s, 50)
plothist(d_200_s, 200)


# ### Q6

# In[16]:

links_pd = pd.read_csv('links.txt', sep=' ', header= None)
links = links_pd.as_matrix()o


# In[ ]:

def RaynorSearch (g,d,k,s,query):
    top_5 = []
    d_new = scaledPR(adj_list, d, k, s)
    d_new = sorted(d_new.items(), key=lambda x: x[1], reverse = True)
    for node in d_new:
        while len(top_5) < 5:
            if query in links[node[0]][1]:
                top_5.append(links[node[0]][1])
    return top_5
RaynorSearch(adj_list, d, 10, 0.85, '34')


# In[ ]:



