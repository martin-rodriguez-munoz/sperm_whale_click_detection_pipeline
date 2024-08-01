import json
import pandas as pd
import numpy as np
import os
import networkx as nx
import scipy.optimize as opt
from scipy.stats import mode
import scipy
from sklearn.cluster import SpectralClustering

def eval_partition(partition_vector,prob_matrix):
    same_matrix = np.matmul(partition_vector.reshape(-1,1),partition_vector.reshape(1,-1)) + np.matmul((1-partition_vector).reshape(-1,1),(1-partition_vector).reshape(1,-1))
    partition_probability_matrix = same_matrix*prob_matrix+(1-same_matrix)*(1-prob_matrix)
    partition_probability_matrix = np.maximum(partition_probability_matrix,1e-15)
    return np.sum(np.log(partition_probability_matrix))

# Cluster coda probabilities using eigenvectors
def eigen_clustering(prob_matrix):
    
    final_output = []
    prob_matrix = (prob_matrix+prob_matrix.transpose())/2

    # Two clicks will not be assigned same coda if there is not path of at least 0.5 probability that connects them
    # Use this to seperate the problem into subcases before we even start 
    G = nx.from_numpy_matrix(prob_matrix > 0.5)
    G = G.to_undirected()
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    to_divide = []

    for subgraph in S:
        selected = np.array(subgraph.nodes)
        sub_prob_matrix = prob_matrix[selected,:][:,selected]
        to_divide.append((sub_prob_matrix,selected))
        
    k = 0
    while k < len(to_divide):
        sub_prob_matrix, original_ids = to_divide[k]
        k += 1
        n = sub_prob_matrix.shape[0] # Number of clicks in case
       
        if n == 1:
            final_output.append(original_ids)
            continue
        
        # Find approximate best 2-cluster clustering
        clustering = SpectralClustering(n_clusters=2,affinity="precomputed").fit(sub_prob_matrix)    
        best_cut_partition_vector = clustering.labels_
        best_cut_score = eval_partition(best_cut_partition_vector,sub_prob_matrix)
        
        # Compare with all clicks in 1 cluster
        no_split_score = eval_partition(np.zeros(n),sub_prob_matrix)
        if no_split_score > best_cut_score:  # If all clicks in 1 cluster is better, resolve subcase with all clicks in 1 coda
            final_output.append(original_ids)
        else: # If the 2 clustering is better split recursivly into 2 subcases
            mask_one = best_cut_partition_vector > 0.5
            mask_two = best_cut_partition_vector <= 0.5
            to_divide.append((sub_prob_matrix[mask_one,:][:,mask_one],original_ids[mask_one]))
            to_divide.append((sub_prob_matrix[mask_two,:][:,mask_two],original_ids[mask_two]))

    
    return final_output
        
# Calculate weight between two codas
# The lower it is the more we want to assign same whale
def compare_clusters(cluster_a,cluster_b,whale_prob_matrix):
    probabilities = []
    for a in cluster_a:
        for b in cluster_b:
            if (a,b) in whale_prob_matrix.keys():
                probabilities.append(whale_prob_matrix[(a,b)])

    if len(probabilities) == 0:
        return 0
    
    probabilities = np.array(probabilities)
    logs = -np.log(probabilities)+np.log(1-probabilities)
    return np.sum(logs)

# Obtain groups from same / different pairs
def find_groups(n,same_different,id_map):
    click_id_to_cluster_id = np.zeros(n)
    all_clusters = {}
    
    last_cluster_id = 0
    for i in range(n):
        click_id_to_cluster_id[i] = last_cluster_id
        all_clusters[last_cluster_id] = [i]
        last_cluster_id += 1

    for i in range(n):
        for j in range(i+1,n,1):
            if same_different[id_map[(i,j)]] > 0.5 and click_id_to_cluster_id[i] != click_id_to_cluster_id[j]:
                # Move all the nodes from j's cluster to i's cluester
                new_cluster_id = click_id_to_cluster_id[i]

                old_cluster_id = click_id_to_cluster_id[j]
                old_cluster = all_clusters[old_cluster_id]
                
                for old_cluster_node in old_cluster:
                    click_id_to_cluster_id[old_cluster_node] = new_cluster_id

                all_clusters[new_cluster_id] += old_cluster
                del all_clusters[old_cluster_id]
    
    return all_clusters

# Finds optimal whale clusterings
def solve_problem(weight_matrix):
    n = weight_matrix.shape[0]

    num_var = int((n*(n-1))/2)
    num_res = int((n*(n-1)*(n-2))/2)

    id_num = 0
    id_map = {}

    c = np.zeros(num_var)

    for i in range(n):
        for j in range(i+1,n,1):
            id_map[(i,j)] = id_num
            c[id_num] = weight_matrix[i][j]
            id_num += 1

    A = np.zeros((num_res,num_var))
    cond_num = 0
    for i in range(n):
        for j in range(i+1,n,1):
            for k in range(j+1,n,1):
                A[cond_num,id_map[(i,j)]] = 2 
                A[cond_num,id_map[(j,k)]] = 2 
                A[cond_num,id_map[(i,k)]] = -2 
                cond_num += 1

                A[cond_num,id_map[(i,j)]] = 2 
                A[cond_num,id_map[(j,k)]] = -2 
                A[cond_num,id_map[(i,k)]] = 2 
                cond_num += 1

                A[cond_num,id_map[(i,j)]] = -2 
                A[cond_num,id_map[(j,k)]] = 2 
                A[cond_num,id_map[(i,k)]] = 2 
                cond_num += 1

    constraint = opt.LinearConstraint(A,ub=3)
    bounds = opt.Bounds(lb=-0.5,ub=1.5)
    ans = opt.milp(c=c,integrality=1,bounds=bounds,constraints=constraint)
    #print(ans["message"])
    return ans, id_map

# Obtains coda and whale clusterings from phase 2 output
def phase_3(click_info_json,prediction_th,input_file_name):
    data = click_info_json

    
    all_times = [float(i) for i in list(data.keys())]
    all_predicted_times  = [i for i in all_times if data[i]["click_probability"] > prediction_th]

    n = len(all_predicted_times)

    coda_prob_matrix = np.zeros((n,n))

     # Calculate same coda probability and label matrix
    for i in range(len(all_predicted_times)):
        i_time = all_predicted_times[i]
        for j in range(len(all_predicted_times)):
            if i == j:
                coda_prob_matrix[i][j] = 1
            else:
                j_time = all_predicted_times[j]
                if j_time in data[i_time]["same_different_times"] and i_time in data[j_time]["same_different_times"]:
                    j_pos = np.where(np.array(data[i_time]["same_different_times"]) == j_time)[0][0]
                    i_pos = np.where(np.array(data[j_time]["same_different_times"]) == i_time)[0][0]

                    coda_prob_matrix[i][j] = data[i_time]["same_different_coda_probability"][j_pos]
                    coda_prob_matrix[j][i] = data[j_time]["same_different_coda_probability"][i_pos]


    # Run coda clustering algorithm
    coda_prob_matrix = (coda_prob_matrix+coda_prob_matrix.transpose())/2
    coda_clusterings = eigen_clustering(coda_prob_matrix)
    
    # Calculate whale weigths
    solo_whale_prob_matrix = {}
    for i in range(len(all_predicted_times)):
        i_time = all_predicted_times[i]
        for j in range(len(all_predicted_times)):
            if i == j:
                solo_whale_prob_matrix[(i,j)] = 1
            else:
                j_time = all_predicted_times[j]
                if j_time in data[i_time]["same_different_times"] and i_time in data[j_time]["same_different_times"]:
                    j_pos = np.where(np.array(data[i_time]["same_different_times"]) == j_time)[0][0]
                    i_pos = np.where(np.array(data[j_time]["same_different_times"]) == i_time)[0][0]

                    solo_whale_prob_matrix[(i,j)] = (data[i_time]["same_different_whale_probability"][j_pos]+data[j_time]["same_different_whale_probability"][i_pos])/2
                    solo_whale_prob_matrix[(j,i)] = (data[j_time]["same_different_whale_probability"][i_pos]+data[i_time]["same_different_whale_probability"][j_pos])/2

    m = len(coda_clusterings)
    #print("m",m)
    whale_weights = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            whale_weights[i][j] = compare_clusters(coda_clusterings[i],coda_clusterings[j],solo_whale_prob_matrix)
            whale_weights[j][i] = whale_weights[i][j]

    # Calculate optimal whale clustering

    # Two codas will not be assigned same whale if there is not path of positive weights that connects them
    # Use this to seperate the problem into subcases before we even start 
    G = nx.from_numpy_matrix(whale_weights < -0.00001)
    G = G.to_undirected()
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    all_predicted_times = np.array(all_predicted_times)
    to_divide = []
    whale_output = []

    for subgraph in S:
        selected = np.array(subgraph.nodes)
        sub_whale_matrix = whale_weights[selected,:][:,selected]
        to_divide.append((sub_whale_matrix,selected))
        
    # Obtain best whale clusterings with linear programming
    sub_problem_pos = 0
    while sub_problem_pos < len(to_divide):
        sub_whale_matrix, original_ids = to_divide[sub_problem_pos]
        n = sub_whale_matrix.shape[0]
        #print("n",n)

        if n > 150:
            print("too many similar codas, splitting arbitarly")
            mask_one = np.arange(n) < n/2
            mask_two = np.arange(n) >= n/2

        
            to_divide.append((sub_whale_matrix[mask_one,:][:,mask_one],original_ids[mask_one]))
            to_divide.append((sub_whale_matrix[mask_two,:][:,mask_two],original_ids[mask_two]))
        else:
            if n > 1:
                ans, id_map = solve_problem(sub_whale_matrix)
                clusters = find_groups(n,ans["x"],id_map)
            else:
                clusters = {}
                clusters[0] = [0]

            for k in clusters.keys():
                mask = np.array([i in clusters[k] for i in range(n)])
                whale_output.append((original_ids[mask]))

        sub_problem_pos += 1

    # Organize output
    coda_unord_whale = []
    unord_whale_id = 0
    for coda_ids in whale_output:
        for coda_id in coda_ids:
            coda_unord_whale.append([all_predicted_times[click_id] for click_id in coda_clusterings[coda_id]]+[unord_whale_id])
        unord_whale_id += 1


    coda_whale = []
    id_map = {}
    next_whale_id = 0
    for time_and_whale in sorted(coda_unord_whale):
        time = time_and_whale[:-1]
        old_whale = time_and_whale[-1]

        if old_whale not in id_map.keys():
            id_map[old_whale] = next_whale_id
            next_whale_id += 1
        coda_whale.append((time,id_map[old_whale]))

    num_codas = len(coda_whale)

    if num_codas == 0:
        final_output_df = pd.DataFrame()

        final_output_df["REC"] = []
        final_output_df["nClicks"] = []
        final_output_df["Duration"] = []
        for i in range(41):
            final_output_df["ICI"+str(i)] = []
        final_output_df["Whale"] = []
        final_output_df["TsTo"] = []
        return final_output_df
    
    coda_info = np.zeros((num_codas,41))

    for coda_id in range(num_codas):
        coda_click_times = np.array(sorted(coda_whale[coda_id][0]))

        ts_to = coda_click_times[0]
        coda_len = min(41,coda_click_times.shape[0])

        coda_icis = coda_click_times[1:] - coda_click_times[:-1]

        coda_info[coda_id][0] = ts_to
        coda_info[coda_id][1:coda_len] = coda_icis[:(coda_len-1)]

    final_output_df = pd.DataFrame()
    final_output_df["REC"] = [input_file_name.split('/')[-1]]*num_codas
    final_output_df["nClicks"] = [len(c[0]) for c in coda_whale]
    final_output_df["Duration"] = [np.max(c[0])-np.min(c[0]) for c in coda_whale]
    for i in range(1,41,1):
        final_output_df["ICI"+str(i)] = coda_info[:,i]
    final_output_df["Whale"] = [c[1] for c in coda_whale]
    final_output_df["TsTo"] = coda_info[:,0]

    return final_output_df

   