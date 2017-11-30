

from IPython.display import HTML

import numpy as np
import pandas as pd
import scipy as sp
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



ccl_pd = pd.read_csv('datasets_498/masterccleids.csv', header=None, sep =",")
ccl = ccl_pd.as_matrix()
ccl_pd.head()


auc_trans_pd = pd.read_csv('datasets_498/auc.csv', names = ccl[0], sep =",")
auc_trans_pd.head()


auc_pd = auc_trans_pd.transpose()
auc = auc_pd.as_matrix()
auc_pd.head()


geneexp_header_pd = pd.read_csv('datasets_498/geneexp_entrezid.csv', header = None, sep =",")
geneexp_header = geneexp_header_pd.as_matrix()
print geneexp_header_pd.shape
geneexp_header_pd.head()



geneexp_values_pd = pd.read_csv('datasets_498/geneexp_values.csv', names = ccl[0], sep =",")
geneexp_values = geneexp_values_pd.as_matrix()
print geneexp_values_pd.shape
geneexp_values_pd.set_index(geneexp_header, inplace=True)
geneexp_values_pd.head()



mutation_header_pd = pd.read_csv('datasets_498/mutation_hugo.csv', header = None, sep = " ")
mutation_header = mutation_header_pd.as_matrix()
print mutation_header_pd.shape
mutation_header_pd.head()



mutation_values_pd = pd.read_csv('datasets_498/mutation_value.csv', names = ccl[0], sep =",")
mutation_values = mutation_values_pd.as_matrix()
print mutation_values_pd.shape
mutation_values_pd = mutation_values_pd.astype('float64')
# mutation_values_pd.set_index(mutation_header, inplace=True)
mutation_values_pd.head()



copynum_header_pd = pd.read_csv('datasets_498/copynum_entrezid.csv', header = None, sep =",")
copynum_header = copynum_header_pd.as_matrix()
print copynum_header_pd.shape
copynum_header_pd.head()



copynum_values_pd = pd.read_csv('datasets_498/copynum_values.csv', names = ccl[0], sep =",")
copynum_values = copynum_values_pd.as_matrix()
print np.shape(copynum_values_pd)
copynum_values_pd.set_index(copynum_header, inplace=True)
copynum_values_pd.head()


exp_cn_trans_pd = pd.concat([copynum_values_pd, geneexp_values_pd])
exp_cn_trans = exp_cn_trans_pd.as_matrix()
print exp_cn_trans_pd.shape
exp_cn_trans_pd.head()



exp_cn_pd = exp_cn_trans_pd.transpose()
exp_cn = exp_cn_pd.as_matrix()


linreg = LinearRegression()
svr_rbf = SVR(kernel = "rbf")
svr_lin = SVR(kernel = "linear")
rf = RandomForestRegressor()


param_lst = {"rf": {"n_estimators": range(6, 11)}, 
             "svr_rbf": {"C": [1, 10, 100, 1000], "gamma": [10**(-6), 10**(-5), 10**(-4)]}, 
             "svr_lin": {"C": [1, 10, 100, 1000], "gamma": [10**(-6), 10**(-5), 10**(-4)]}}
algo_lst = ["rf", "svr_rbf", "svr_lin"]


# In[322]:

num_trials = 1
r2_rf, r2_svr_rbf, r2_svr_lin = [], [], []
for outerMCCV in range(num_trials):
    out_x_train, out_x_test, out_y_train, out_y_test = train_test_split(exp_cn, auc, test_size=0.2, random_state=42)
    out_y_train = out_y_train.flatten()
    out_y_test = out_y_test.flatten()
    param_outer, score_outer = {"rf": [], "svr_rbf": [], "svr_lin": []}, {"rf": [], "svr_rbf": [], "svr_lin": []}
    param_inner, score_inner = {"rf": [], "svr_rbf": [], "svr_lin": []}, {"rf": [], "svr_rbf": [], "svr_lin": []}
    for innerMCCV in range(num_trials):
        for algo in algo_lst:
            if algo == "rf":
                clf = RandomForestRegressor()
            if algo == "svr_rbf":
                clf = SVR(kernel = "rbf")
            if algo == "svr_lin":
                clf = SVR(kernel = "linear")
            grid = GridSearchCV(clf, param_grid=param_lst[algo], cv=5)
            grid.fit(out_x_train, out_y_train)
            results = grid.cv_results_
            best_fit = np.argmax(results.get("mean_test_score"))
            r2 = results.get("mean_test_score")
            get_params = results.get("params")[best_fit]
            param_inner[algo].append([get_params])
            score_inner[algo].append([r2])
            print algo, r2
    for algo in algo_lst:
        score_outer[algo] = list(map(lambda x: np.mean(x), score_inner[algo]))
    print score_outer
    for algo in algo_lst:
        best_case = np.argmax(score_outer[algo])
        param_outer[algo] = list(map(lambda x: x, param_inner[algo][best_case]))
        print param_outer
        if algo == "rf":
            clf = RandomForestRegressor(n_estimators = param_outer[algo][0]["n_estimators"])
        if algo == "svr_rbf":
            clf = SVR(kernel = "rbf", C = param_outer[algo][0]["C"], gamma = param_outer[algo][0]["gamma"])
        if algo == "svr_lin":
            clf = SVR(kernel = "linear", C = param_outer[algo][0]["C"], gamma = param_outer[algo][0]["gamma"])
        clf.fit(out_x_train, out_y_train)
        r2 = clf.score(out_x_test, out_y_test)
        if algo == "rf":
            r2_rf.append(r2)
        if algo == "svr_rbf":
            r2_svr_rbf.append(r2)
        if algo == "svr_lin":
            r2_svr_lin.append(r2)
print "rf :", np.average(r2_rf)
print "svr rbf :", np.average(r2_svr_rbf)
print "svr lin :", np.average(r2_svr_lin)


# In[ ]:



