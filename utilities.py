import datetime
import numpy as np
import pickle
import gzip
import argparse
from scipy import optimize

def log(str, logfile=None):
    str = '[{datetime.datetime.now()}] {str}'.format(datetime.datetime.now(), str)
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

# %% This function generates input matrix for WDP class from binary data txt
# bianry_file_path: binary data txt
# exapmle:
# bid = bid_recover('data/....txt')
# X = WDP(bid)  # initialize WDP class
# X.initialize_mip(verbose=True)   # initialize MIP
# X.solve_mip()   # solve MIP
# X.print_optimal_allocation()
def bid_recover(bianry_file_path, unit_file_path=''):
    bid_load = np.loadtxt(bianry_file_path)
    bid_recover = []
    for i in range(0,bid_load.shape[0]):
        bid_recover.append(bid_load[i].reshape(1,-1))
    if unit_file_path=='':
        unit_load = np.ones((1,bid_load.shape[1]-1))
    else:
        unit_load = np.loadtxt(unit_file_path).reshape(1,-1)
    bid_recover.append(unit_load.reshape(1,-1))

    return bid_recover

# %% This function checks whether seed is a valid random seed or not.
def valid_seed(seed):
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed

# %% This function checks whether probability is valid or not.
def valid_prob(prob):
    prob = int(prob)
    if prob < 0 or prob > 100:
        raise argparse.ArgumentTypeError(
                "probability must be any integer between 0 and 100 inclusive")
    return prob

def linear_solve(instance_matrix,units_matrix):
    #objective function
    c = instance_matrix[:,-1]
    num_bid = len(c)
    #constraints left parameters and right parameters
    a = np.transpose(instance_matrix[:,0:-1])
    b = units_matrix


    bound = []
    for i in range(num_bid):
        bound.append((0,1))

    res=optimize.linprog(-c,A_ub=a,b_ub=b,bounds=tuple(bound))

    return res.x

def dual_linear_solve(instance_matrix,units_matrix):
    #objective function
    c = instance_matrix[:,-1]
    #constraints left parameters and right parameters
    a = instance_matrix[:,0:-1]
    b = units_matrix
    res=optimize.linprog(b,A_ub=-a,b_ub=-c)

    return res.x
    
# %% This function compute a bipartite graph representation of the WDP. 
# In this representation, the items and bids of the WDP are the
# left- and right-hand side nodes, and an edge links two nodes iff the
# item is included in the bid. Both the nodes and edges carry features.
# bianry_file_path: binary data txt
# unit_file_path: for multi-ca unit numer txt
# index_flag: whether the index of bid needs to be used as a feature
# bid_features : dictionary of type {'names': list, 'values': np.ndarray}
# edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
# item_features : dictionary of type {'names': list, 'values': np.ndarray}
def extract_feature(bianry_file_path, unit_file_path='', index_flag=False):
    # Matrix to show the relationship between items and bids
    # num_items*num_bids
    relation = np.transpose(np.delete(np.loadtxt(bianry_file_path),-1, axis = 1))


    # compute feature for items
    # 01 The number of units an item are required
    item_required = np.sum(relation,axis=1).reshape(-1,1)
    # 02 How many units of an item has
    if unit_file_path=='':
        item_provided = np.ones(item_required.shape[0]).reshape(-1,1)
    else:
        # multi-unit CA
        # need to read the information from file
        item_provided = np.loadtxt(unit_file_path).reshape(-1,1)

    item_feat_vals = np.concatenate((item_required,item_provided), axis=-1)
    item_feat_names = list(range(1,item_required.shape[0]+1))

    item_features = {
        'names': item_feat_names,
        'values': item_feat_vals,}

    # compute feature for bids
    # 01 A bid requires how many units
    bid_size = np.sum(relation,axis=0).reshape(-1,1)
    # 02 The price of this bid
    bid_price = np.loadtxt(bianry_file_path)[:,-1].reshape(-1,1)
    # 03 The index of this bid
    bid_index = np.array(range(0,bid_size.shape[0])).reshape(-1,1)

    if index_flag==True:
        bid_feat_vals = np.concatenate((bid_size,bid_price,bid_index), axis=-1)
    else:
        bid_feat_vals = np.concatenate((bid_size,bid_price), axis=-1)
    bid_feat_names = list(range(1,bid_size.shape[0]+1))

    bid_features = {
        'names': bid_feat_names,
        'values': bid_feat_vals,}

    # compute indices and feature for edge
    index = np.where(relation!=0)
    edge_feat_indices = np.insert(index[0].reshape(1,-1), 1, values=index[1].reshape(1,-1), axis=0)

    # 01 A bid requires how many units of an items
    edge_feat_vals = np.zeros((edge_feat_indices.shape[1],1))
    for i in range(edge_feat_indices.shape[1]):
        edge_feat_vals[i] = relation[edge_feat_indices[0,i],edge_feat_indices[1,i]]
    
    edge_feat_names = list(range(1,edge_feat_indices.shape[1]+1))

    edge_features = {
        'names': edge_feat_names,
        'indices': edge_feat_indices,
        'values': edge_feat_vals,}

    return item_features, edge_features, bid_features












