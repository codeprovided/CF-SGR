import numpy as np
import pandas as pd
import scipy.sparse as sp
def generate_G_from_H(H):
    """ carry the hypergraph symmetry change """
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.array(H.sum(1))
    # the degree of the hyperedge
    DE = np.array(H.sum(0))

    invDE2 = sp.diags(np.power(DE, -0.5).flatten())
    DV2 = sp.diags(np.power(DV, -0.5).flatten())
    W = sp.diags(W)
    HT = H.T

    invDE_HT_DV2 = invDE2 * HT * DV2
    G = DV2 * H * W * invDE2 * invDE_HT_DV2
    return G, invDE_HT_DV2



def generate_subgraph_mapping(interacted_data, time_data, timestamp, gu_flag):
    """
    generate the hypergraph subgraphs from raw user/group-item-time interactions according to the time frame
    :gu_flag: 0:user, 1:group
    """
    subgraphs = {}
    subgraph_G = {}
    subgraph_mapping_u = {}
    subgraph_mapping_i = {}
    for t in timestamp:
        if gu_flag == 0:
            subgraphs[t] = {'u': [], 'i': []}
        else:
            subgraphs[t] = {'g': [], 'i': []}
        subgraph_mapping_u[t] = {}
        subgraph_mapping_i[t] = {}
        subgraph_G[t] = {}
    for u in interacted_data:
        ilist = interacted_data[u]
        tlist = time_data[u]
        for i, t in zip(ilist, tlist):
            # encode the reorder number into mapping matrix
            if i not in subgraph_mapping_i[t]:
                subgraph_mapping_i[t][i] = len(subgraph_mapping_i[t])
            if u not in subgraph_mapping_u[t]:
                subgraph_mapping_u[t][u] = len(subgraph_mapping_u[t])
            subgraphs[t][list(subgraphs[t].keys())[0]].append(subgraph_mapping_u[t][u])
            subgraphs[t]['i'].append(subgraph_mapping_i[t][i])

    # generate the subgraphs sparse matrix
    for time in timestamp:
        if gu_flag == 0:
            col = subgraphs[time]['u']
        else:
            col = subgraphs[time]['g']
        row = subgraphs[time]['i']
        data = np.ones(len(row))
        sp_mat = sp.coo_matrix((data, (row, col)), shape=(len(subgraph_mapping_i[time]), len(subgraph_mapping_u[time])))
        subgraph_G[time]['G'], subgraph_G[time]['E'] = generate_G_from_H(sp_mat)
        print(f"has generate the subgraph: {time}")
        if gu_flag == 0:
            print(f"the reorder items and users number:\n item:{len(subgraph_mapping_i[time])} and user:{len(subgraph_mapping_u[time])}")
        else:
            print(
                f"the reorder items and groups number:\n item:{len(subgraph_mapping_i[time])} and group:{len(subgraph_mapping_u[time])}")

    return subgraph_mapping_i, subgraph_mapping_u, subgraph_G


def generate_subgraph_mapping_dynamic(subgraph_mapping_i, item_num):
    """ generate the dynamic subgraph mapping dicts """

    time_list = sorted(list(subgraph_mapping_i.keys()))
    reversed_subgraph_mapping_i = {}
    for t in subgraph_mapping_i:
        reversed_subgraph_mapping_i[t] = [0]*len(subgraph_mapping_i[t])
        for i in subgraph_mapping_i[t]:
            reversed_subgraph_mapping_i[t][subgraph_mapping_i[t][i]] = i
    if 0 not in subgraph_mapping_i:
        subgraph_mapping_i[0] = {}
        for i in range(item_num):
            subgraph_mapping_i[0][i] = i
    index_offset = item_num
    offset_record = {}
    for t in subgraph_mapping_i:
        if t == 0:
            continue
        else:
            offset_record[t] = index_offset
            for i in subgraph_mapping_i[t]:
                subgraph_mapping_i[t][i] += index_offset
            index_offset += len(subgraph_mapping_i[t])

    # store the item i's dynamic index in each timestamp
    subgraph_sequence_i = {}
    # sugraph_sequence_i[i] = [i,i,i,i,i,5,6,7,8,9,10,...,15,...]
    for i in range(item_num):
        subgraph_sequence_i[i] = np.array([i]*(3 + max(time_list)))

    for t in time_list:
        for i in subgraph_mapping_i[t]:
            subgraph_sequence_i[i][t+1:] = subgraph_mapping_i[t][i]

    # generate the latest reversed mapping dict
    reversed_subgraph_mapping_latest_i = {}
    for t in subgraph_mapping_i:
        if t == 0:
            continue
        reversed_subgraph_mapping_latest_i[t] = [0]*len(subgraph_mapping_i[t])
        for i in subgraph_mapping_i[t]:
            original_item_idx = subgraph_mapping_i[t][i] - offset_record[t]
            reversed_subgraph_mapping_latest_i[t][original_item_idx] = subgraph_sequence_i[i][t]
    item_dy_num = index_offset
    return subgraph_mapping_i, reversed_subgraph_mapping_i, subgraph_sequence_i, \
           reversed_subgraph_mapping_latest_i, time_list, item_dy_num
