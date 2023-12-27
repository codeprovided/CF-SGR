import torch
import random
import os
import numpy as np
import math
import pandas as pd
from collections import defaultdict


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def matrix_to_sp_tensor(H):
    H_coo = H.tocoo()
    indices = []
    # transpose the row and column
    indices.append(H_coo.row)
    indices.append(H_coo.col)
    indices = torch.LongTensor(np.array(indices))
    values = torch.from_numpy(H_coo.data)
    tensor_sparse = torch.sparse_coo_tensor(indices=indices, values=values, size=H_coo.shape, dtype=torch.float)
    return tensor_sparse


def check_save_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"set the save_paths:{path}")


def Person(vector1, vector2):
    n = len(vector1)
    sum1 = sum(float(vector1[i]) for i in range(n))


def generate_dynamic_index(user_pair_set, max_sequence_length, subgraph_sequence_i, subgraph_sequence_u,
                           subgraph_mapping_i, subgraph_mapping_u, u, data_type='train'):
    assert data_type in {'train', 'target'}
    time_temp = user_pair_set[2][-1]
    temp = []
    tempu = []
    for ii, tt in zip(user_pair_set[1][:max_sequence_length], user_pair_set[2][:max_sequence_length]):
        if tt == time_temp:
            temp.append(subgraph_sequence_i[ii][tt])
            tempu.append(subgraph_sequence_u[u][tt])
        else:
            temp.append(subgraph_mapping_i[tt][ii])
            tempu.append(subgraph_mapping_u[tt][u])
    if data_type == 'train':
        return temp, tempu
    elif data_type == 'target':
        return temp
    else:
        raise ValueError


def bayesian_pairwise_loss(pos_logits, neg_logits, istarget):
    return torch.sum(
        - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
        torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
    ) / torch.sum(istarget)


def load_gu_dict(inputpath):
    df_gu = pd.DataFrame(pd.read_csv(inputpath))
    gu_dict = defaultdict(list)
    df_gu = df_gu.groupby(by='group')
    for _, df in df_gu:
        gu_dict[int(df['group'].tolist()[0])] = sorted(df['user'].tolist())
    return gu_dict


def calculate_evaluate_metric(args, pred_list, k_list, HT, NDCG, user_list, type_m):
    valid_user = 0
    for u in user_list:
        u_pred_list = -pred_list[u - 1]
        rank = u_pred_list.argsort().argsort()[0]  # the first one is the target item
        valid_user += 1
        for k in range(len(k_list)):
            if rank < k_list[k]:
                HT[k] += 1
                NDCG[k] += 1.0 / np.log2(rank + 2)
    for k in range(len(k_list)):
        HT[k] = HT[k] / valid_user
        NDCG[k] = NDCG[k] / valid_user
    user_test_res = {
        "test_type": '{:}'.format(type_m),
        "HT@5": '{:.4f}'.format(HT[0]), "NDCG@5": '{:.4f}'.format(NDCG[0]),
        "HT@10": '{:.4f}'.format(HT[1]), "NDCG@10": '{:.4f}'.format(NDCG[1]),
        "HT@20": '{:.4f}'.format(HT[2]), "NDCG@20": '{:.4f}'.format(NDCG[2]),
        "HT@50": '{:.4f}'.format(HT[3]), "NDCG@50": '{:.4f}'.format(NDCG[3])
    }
    print("the test data result:")
    print(user_test_res)
    with open(f"{os.path.join(args.test_res_log_file, args.dataset_name)}_res.log", 'a') as f:
        f.write(str(user_test_res) + '\n')
    f.close()

    return HT, NDCG


def set_ndcg_bins(pred_list, k_list, user_list, group_bins, g_ndcg_bins):
    valid_user = 0
    for u in user_list:
        u_pred_list = -pred_list[u - 1]
        rank = u_pred_list.argsort().argsort()[0]  # the first one is the target item
        valid_user += 1
        for size, groups in group_bins.items():
            if valid_user in groups:
                tmp_size = size
                break

        for k in range(len(k_list)):
            if rank < k_list[k]:
                # put the target group ndcg into bins
                g_ndcg_bins[tmp_size][k] += 1.0 / np.log2(rank + 2)

    for size in group_bins.keys():
        length = len(group_bins[size])
        for k in range(len(k_list)):
            g_ndcg_bins[size][k] /= length
    return g_ndcg_bins
