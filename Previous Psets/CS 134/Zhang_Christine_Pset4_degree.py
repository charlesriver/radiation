
# coding: utf-8

# In[342]:

import numpy as np
import pandas as pd


# In[347]:

slash_pd = pd.read_csv('soc-Slashdot0811.txt', skiprows = [0,1,2], sep='\t')
enron_pd = pd.read_csv('email-enron.txt', skiprows = [0,1,2], sep='\t')
epinions_pd = pd.read_csv('soc-Epinions1.txt', skiprows = [0,1,2], sep='\t')
wiki_pd = pd.read_csv('wiki-Talk.txt', skiprows = [0,1,2], sep='\t')
dblp_pd = pd.read_csv('com-dblp.ungraph.txt', skiprows = [0,1,2], sep='\t')
lj_pd = pd.read_csv('com-lj.ungraph.txt', skiprows = [0,1,2], sep='\t')
youtube_pd = pd.read_csv('com-youtube.ungraph.txt', skiprows = [0,1,2], sep='\t')
orkut_pd = pd.read_csv('com-orkut.ungraph.txt', skiprows = [0,1,2], sep='\t')


# In[356]:

def combined (network_pd, directed_or_no):
    
    network = network_pd.as_matrix()
    max_num = 10000
    network = network[np.logical_and(network[:,0] < max_num, network[:,1] < max_num)]
    
    def neighbors (network, directed_or_no):
        d = dict()
        for i,j in enumerate(network[:,0]):
            if j in d:
                d[j].append(network[i,1])
                if directed_or_no == True:
                    if network[i,1] in d:
                        d[network[i,1]].append(j)
                    else:
                        d[network[i,1]] = [j]
            else:
                d[j] = [network[i,1]]
        return d
    
    d_1 = neighbors (network, directed_or_no)

    def create_dict (network):
        d_2 = dict((x, network.count(x)) for x in set(network))
        return d_2
    
    d_2 = create_dict (list(network[:,0]))
    
    def degree (node):
        try:
            return d_2[node]
        except Exception:
            return 0
    
    def numerator (network):
        total_sum = 0
        for each in set(network[:,0]):
            total_sum += degree (each)
        return float(total_sum) 
    
    def denominator (network):
        sum_total = 0
        for each_node in d_1:
            sum_inner = 0
            for each in d_1[each_node]:
                sum_inner += degree(each)
            sum_total += sum_inner / float(len(d_1[each_node]))
        return sum_total 

    print numerator(network) / denominator (network)


# In[357]:

combined(slash_pd, False)
combined(enron_pd, False)
combined(epinions_pd, False)
combined(wiki_pd, False)


# In[358]:

combined (dblp_pd, True)
combined (lj_pd, True)
combined (youtube_pd, True)
combined (orkut_pd, True)


# In[ ]:



