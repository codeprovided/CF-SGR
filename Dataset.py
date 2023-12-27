import random
import argparse
import torch
import numpy as np
import pandas as pd
import torch
import os
import datetime
import scipy.sparse as sp
from torch.utils import data
from collections import defaultdict
from utils.common_utils import generate_dynamic_index


def data_partition(args):
    """
    load the raw ui and gi interaction data from file and divide them into train/valid/test dataset
    divided data set:
    1. user_train: just use in model train
    2. user_test: just use in model pretrain
    """
    # for ui interactions
    User = defaultdict(list)
    User_time = defaultdict(list)
    user_train = {}
    user_test = {}
    user_train_time = {}
    user_test_time = {}
    neg_test = {}
    time_set_train = set()
    time_set_test = set()
    user_num = 0
    item_num = 0
    g_item_num = 0
    # for gi interactions
    Group = defaultdict(list)
    Group_time = defaultdict(list)
    group_train = {}
    group_test = {}
    group_train_time = {}
    group_test_time = {}
    group_neg_test = {}
    group_time_set_train = set()
    group_time_set_test = set()
    group_num = 0

    dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    df_ui = pd.DataFrame(pd.read_csv(dataset_dir + f"/{args.dataset_name}_ui.csv"))
    df_gi = pd.DataFrame(pd.read_csv(dataset_dir + f"/{args.dataset_name}_gi.csv"))
    df_ui['timestamp'] = pd.to_datetime(df_ui['timestamp'])
    if args.dataset_name == 'Yelp':
        # for yelp
        t_map = {2019: [1, 2, 3], 2020: [4, 5, 6], 2021: [7, 8, 9], 2022: [10, 11, 12]}
        m_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 2}
    else:
        # for amazon
        t_map = {2000: [1], 2001: [1], 2002: [1], 2003: [1], 2004: [1], 2005: [1], 2006: [1], 2007: [1], 2008: [1],
                 2009: [2, 3], 2010: [4, 5], 2011: [6, 7], 2012: [8, 9], 2013: [10, 11], 2014: [12, 13], \
                 2015: [14, 15], 2016: [16, 17], 2017: [18, 19], 2018: [20]}
        m_map = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1}
    user_list = df_ui['user'].unique().tolist()
    group_list = df_gi['group'].unique().tolist()
    for u in user_list:
        # change timestamp into time map
        timestamp_list = df_ui[df_ui['user'] == u]['timestamp'].tolist()
        item_list = df_ui[df_ui['user'] == u]['item'].tolist()
        assert len(timestamp_list) == len(item_list)
        for i in range(len(timestamp_list)):
            year = int(timestamp_list[i].year)
            month = int(timestamp_list[i].month)
            temp_map = t_map[year]
            if len(temp_map) == 1:
                User_time[u].append(temp_map[0])
            else:
                User_time[u].append(temp_map[m_map[month]])
            User[u].append(item_list[i])
            user_num = max(u, user_num)
            item_num = max(item_num, item_list[i])
    for g in group_list:
        # change timestamp into time map
        g_timestamp_list = df_gi[df_gi['group'] == g]['timestamp'].tolist()
        g_item_list = df_gi[df_gi['group'] == g]['item'].tolist()
        assert len(g_timestamp_list) == len(g_item_list)
        for i in range(len(g_timestamp_list)):
            year = int(g_timestamp_list[i])
            temp_map = t_map[year]
            Group_time[g].append(temp_map[0])
            Group[g].append(g_item_list[i])
            group_num = max(g, group_num)
            g_item_num = max(g_item_num, g_item_list[i])
    assert item_num >= g_item_num
    neg_test[0] = [0] * (args.i_neg_sample_size + 1)
    group_neg_test[0] = [0] * (args.i_neg_sample_size + 1)
    # divide the user_list into train/valid/test
    # items: [1,2,3,4,5,6]
    # user_train: [1,2,3,4,5]
    # user_test: [1,2,3,4,5,6]
    for user in User:
        items_length = len(User[user])
        if items_length < 3:  # for the ui interaction length <3, only use in train data
            user_train[user] = User[user]
            user_test[user] = User[user]
            user_train_time[user] = User_time[user]
            user_test_time[user] = User_time[user]
            neg_test[user] = [User[user][-1]]
        else:
            user_train[user] = User[user][:-1]
            user_test[user] = User[user]
            user_train_time[user] = User_time[user][:-1]
            user_test_time[user] = User_time[user]
            time_set_train.update(user_train_time[user])
            time_set_test.update(user_test_time[user])
            neg_test[user] = [User[user][-1]]

    # divide the group_list into train/valid/test
    for group in Group:
        items_length = len(Group[group])
        if items_length < 3:
            group_train[group] = Group[group]
            group_test[group] = Group[group]
            group_train_time[group] = Group_time[group]
            group_test_time[group] = Group_time[group]
            group_neg_test[group] = [Group[group][-1]]
        else:
            group_train[group] = Group[group][:-1]
            group_test[group] = Group[group]
            group_train_time[group] = Group_time[group][:-1]
            group_test_time[group] = Group_time[group]
            group_time_set_train.update(group_train_time[group])
            group_time_set_test.update(group_test_time[group])
            group_neg_test[group] = [Group[group][-1]]

    # generate negative samples per item in ui and gi interactions
    for u in user_list:
        for i in range(args.i_neg_sample_size):
            neg_i = random.randint(1, item_num - 1)
            while neg_i == User[u][-1]:
                neg_i = random.randint(1, item_num - 1)
            neg_test[u].append(neg_i)  # [u_num 100]
    for g in group_list:
        for i in range(args.i_neg_sample_size):
            neg_i = random.randint(1, g_item_num - 1)
            while neg_i == Group[g][-1]:
                neg_i = random.randint(1, g_item_num - 1)
            group_neg_test[g].append(neg_i)  # [g_num 100]
    # all_rank negative sample, select the whole item set as the test samples
    # init the zero line with zero vector
    # neg_test[0] = [0] * item_num
    # for u in user_list:
    #     pos_item = neg_test[u]
    #     item_set = [pos_item]
    #     neg_set = [i for i in range(1, item_num + 1) if i != pos_item]
    #     item_set += neg_set
    #     neg_test[u] = item_set  # [u_num I]
    #
    # group_neg_test[0] = [0] * g_item_num
    # for g in group_list:
    #     pos_item = group_neg_test[g]
    #     item_set = [pos_item]
    #     neg_set = [i for i in range(1, g_item_num + 1) if i != pos_item]
    #     item_set += neg_set
    #     group_neg_test[g] = item_set  # [g_num I]

    user_res = (user_train, user_test, (user_train_time, user_test_time, time_set_train, time_set_test), \
                neg_test, item_num + 1, user_num + 1)
    group_res = (
        group_train, group_test, (group_train_time, group_test_time, group_time_set_train, group_time_set_test), \
        group_neg_test, g_item_num + 1, group_num + 1)
    return user_res, group_res


def generate_pos_batch(user_train, user_train_time, users_list, subgraph_mapping_i, subgraph_mapping_u, \
                       subgraph_sequence_i, subgraph_sequence_u, max_sequence_length=50, data_flag='user'):
    """
    generate positive batch data into dataloader
    :return:train_batch_set, target_batch_set, test_batch_set
    """

    def _slide_windows(uid, items_list, times_list, max_sequence_length):
        pad_length = max_sequence_length - len(items_list)
        if pad_length <= 0:
            return (uid, items_list[:max_sequence_length], times_list[:max_sequence_length])
        else:
            return (uid, list(np.pad(items_list, (pad_length, 0))), list(np.pad(times_list, (pad_length, 0))))

    def _generate_sequences_padding(user_list, user_items_list, user_times_list, max_sequence_length):
        sequence_list = {}
        for u in user_list:
            sequence_list[u] = []
            temp_idx = 1
            assert len(user_items_list[u]) == len(user_times_list[u])
            st_idx = 0
            ed_idx = len(user_items_list[u])
            while temp_idx < ed_idx:
                temp_idx += 1
                if temp_idx - st_idx > max_sequence_length:
                    st_idx = temp_idx - max_sequence_length
                sequence_list[u].append(_slide_windows(u, user_items_list[u][st_idx:temp_idx], \
                                                       user_times_list[u][st_idx:temp_idx], max_sequence_length))
        return sequence_list

    assert len(users_list) == max(users_list), "throw exception cause the missing user in users_list"
    # draw the interactions from user-item-time sequences
    user_items_list = user_train
    user_times_list = user_train_time
    user_sequence_list = _generate_sequences_padding(users_list, \
                                                     user_items_list, user_times_list,
                                                     max_sequence_length + 1)
    sequences_number = sum([len(user_sequence_list[i]) for i in user_sequence_list.keys()])
    train_seq_number = sequences_number
    # for train
    train_sequence_track = np.zeros((train_seq_number, max_sequence_length), dtype=np.int64)
    train_sequence_original = np.zeros((train_seq_number, max_sequence_length), dtype=np.int64)
    train_sequence_user_track = np.zeros((train_seq_number, max_sequence_length), dtype=np.int64)
    train_users = np.zeros(train_seq_number, dtype=np.int64)
    # for test
    test_sequence_track = np.zeros((len(users_list) + 1, max_sequence_length), dtype=np.int64)
    test_sequence_original = np.zeros((len(users_list) + 1, max_sequence_length), dtype=np.int64)
    test_sequence_user_track = np.zeros((len(users_list) + 1, max_sequence_length), dtype=np.int64)
    test_users = np.zeros(len(users_list) + 1, dtype=np.int64)
    # for target
    target_sequence_original = np.zeros((train_seq_number, max_sequence_length), dtype=np.int64)
    train_seq_idx = 0
    for u in users_list:
        last_i_idx = len(user_sequence_list[u]) - 1
        for i in range(len(user_sequence_list[u])):
            user_pair_set = user_sequence_list[u][i]
            if data_flag == 'pretrain':
                if i == last_i_idx:
                    # generate u's test sequence
                    test_sequence_original[u][:] = user_pair_set[1][-max_sequence_length:]
                    test_users[u] = u
                # generate target sequence
                target_sequence_original[train_seq_idx] = user_pair_set[1][1:max_sequence_length + 1]
                # generate train sequence
                train_sequence_original[train_seq_idx][:] = user_pair_set[1][:max_sequence_length]
                train_users[train_seq_idx] = u
                train_seq_idx += 1
            else:
                # user_pair_set: (uid, item_sequence, time_sequence)
                if i == last_i_idx:
                    # generate the u's test sequence
                    # (uid, uid_items_sequence, uid_times_sequence)
                    test_sequence_track[u][:] = [subgraph_mapping_i[tt][ii] for ii, tt in
                                                 zip(user_pair_set[1][-max_sequence_length:],
                                                     user_pair_set[2][-max_sequence_length:])]
                    test_sequence_original[u][:] = user_pair_set[1][-max_sequence_length:]
                    test_sequence_user_track[u][:] = [subgraph_mapping_u[t][u] for t in
                                                      user_pair_set[2][-max_sequence_length:]]
                    test_users[u] = u
                # generate target sequence
                target_sequence_original[train_seq_idx] = user_pair_set[1][1:max_sequence_length + 1]
                # generate train sequence
                temp, tempu = generate_dynamic_index(user_pair_set, max_sequence_length,
                                                     subgraph_sequence_i, subgraph_sequence_u,
                                                     subgraph_mapping_i, subgraph_mapping_u, u,
                                                     'train')
                train_sequence_track[train_seq_idx][:] = temp
                train_sequence_original[train_seq_idx][:] = user_pair_set[1][:max_sequence_length]
                train_sequence_user_track[train_seq_idx][:] = tempu
                train_users[train_seq_idx] = u
                train_seq_idx += 1
    if data_flag == 'pretrain':
        return (train_sequence_original, train_users), \
               target_sequence_original, \
               (test_sequence_original, test_users)
    else:
        return (train_sequence_track, train_sequence_original, train_sequence_user_track, train_users), \
               target_sequence_original, \
               (test_sequence_track, test_sequence_original, test_sequence_user_track)


class PretrainDataset(data.Dataset):
    """ pretrain user-item dataset """

    def __init__(self, subgraph_mapping_i, subgraph_mapping_u, subgraph_sequence_i, subgraph_sequence_u,
                 users_list, items_list, pretrain_items, pretrain_times, args):
        self.args = args
        self.pretrain_items = pretrain_items
        self.pretrain_times = pretrain_times
        self.items_list = items_list
        self.data_dir = os.path.join(args.data_dir, args.dataset_name, "{}_ui.csv".format(args.dataset_name))
        self.pre_train_batch_set, self.pre_target_batch_set, self.pre_test_batch_set = \
            generate_pos_batch(self.pretrain_items, self.pretrain_times, users_list, subgraph_mapping_i,
                               subgraph_mapping_u, subgraph_sequence_i, subgraph_sequence_u, args.max_sequence_length,
                               'pretrain')

    def __len__(self):
        return len(self.pre_train_batch_set[0])

    def __getitem__(self, index):
        """
        batch sample: index, pre_train_sequence, target_item, target_neg_item_list
         """
        pre_train_sequence = self.pre_train_batch_set[0][index]
        pre_train_user = self.pre_train_batch_set[1][index]
        labels = self.pre_target_batch_set[index]
        cur_tensors = (
            torch.tensor(index, dtype=torch.long),
            torch.tensor(pre_train_user, dtype=torch.long),
            torch.tensor(pre_train_sequence, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )
        return cur_tensors


class TestPreTrainDataset(data.Dataset):
    """ load test user-item interactions for pretrain """

    def __init__(self, pre_test_batch_set, ui_neg_test):
        self.test_sequence_original = pre_test_batch_set[0]
        self.test_users = pre_test_batch_set[1]
        self.ui_neg_test = ui_neg_test

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, index):
        """ batch: index, test_user, test_sequence_ori, labels """
        cur_tensors = (
            torch.tensor(index, dtype=torch.long),
            torch.tensor(self.test_users[index], dtype=torch.long),
            torch.tensor(self.test_sequence_original[index], dtype=torch.long),
            torch.tensor(self.ui_neg_test[index], dtype=torch.long)
        )
        return cur_tensors


class TrainUserDataset(data.Dataset):
    """ load user-item interactions dataset """

    def __init__(self, subgraph_mapping_i, subgraph_mapping_u, subgraph_sequence_i, subgraph_sequence_u, users_list,
                 user_train, user_train_time, item_att_dict, args):
        self.users_list = users_list
        self.item_att_dict = item_att_dict
        self.args = args
        self.train_batch_set, self.target_batch_set, self.test_batch_set = \
            generate_pos_batch(user_train, user_train_time, self.users_list, subgraph_mapping_i, subgraph_mapping_u,
                               subgraph_sequence_i, subgraph_sequence_u, args.max_sequence_length, 'user')

    def __len__(self):
        return len(self.train_batch_set[-1])

    def __getitem__(self, index):
        """
        sample batch: (seq_id, train_sequences_list, train_sequences_dy_list, train_uid_track,
        labels, items_attributes)
        """
        item_attributes = []
        seq_idx = index
        train_sequences_list = self.train_batch_set[1][seq_idx]
        train_sequences_dy_list = self.train_batch_set[0][seq_idx]  # [B L]
        train_uid_track = self.train_batch_set[2][seq_idx]  # [B L]
        labels = self.target_batch_set[seq_idx]
        # generate attribute padding
        for i in train_sequences_list:
            item_attribute = [0] * self.args.attributes_num
            for att in self.item_att_dict[str(i)]:
                item_attribute[int(att)] = 1
            item_attributes.append(item_attribute)  # [L att_size]
        cur_tensors = (
            torch.tensor(seq_idx, dtype=torch.long),
            torch.tensor(train_sequences_list, dtype=torch.long),
            torch.tensor(train_sequences_dy_list, dtype=torch.long),
            torch.tensor(train_uid_track, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(item_attributes, dtype=torch.long)
        )
        return cur_tensors


class EvalUserDataset(data.Dataset):
    def __init__(self, user_list, test_batch_set, ui_neg_test, ui_neg_test_dy):
        self.user_list = user_list
        self.test_batch_set = test_batch_set
        self.ui_neg_test = ui_neg_test
        self.ui_neg_test_dy = ui_neg_test_dy

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, uidx):
        """ sample batch:(uid, test_seq_ori, test_seq_dy, user_dy, labels) """
        user = self.user_list[uidx]
        test_sequences_dy = self.test_batch_set[0][user]
        test_sequences_ori = self.test_batch_set[1][user]
        test_user_dy = self.test_batch_set[2][user]
        labels = self.ui_neg_test[user]  # [100]
        cur_tensors = (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(test_sequences_ori, dtype=torch.long),
            torch.tensor(test_sequences_dy, dtype=torch.long),
            torch.tensor(test_user_dy, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
        return cur_tensors


class TrainGroupDataset(data.Dataset):
    def __init__(self, subgraph_mapping_g_i, subgraph_mapping_g, subgraph_sequence_g_i, subgraph_sequence_g,
                 groups_list, group_train, group_train_time, item_att_dict, args):
        self.item_att_dict = item_att_dict
        self.subgraph_sequence_g_i = subgraph_sequence_g_i
        self.args = args
        self.train_batch_set, self.target_batch_set, self.test_batch_set = \
            generate_pos_batch(group_train, group_train_time, groups_list, subgraph_mapping_g_i, subgraph_mapping_g,
                               subgraph_sequence_g_i, subgraph_sequence_g, args.max_sequence_length, 'user')

    def __len__(self):
        return len(self.train_batch_set[-1])

    def __getitem__(self, index):
        """ sample batch:(seq_id, train_sequences_list, train_sequences_dy_list, train_uid_track,
        labels, items_attributes)
          """
        item_attributes = []
        seq_idx = index
        train_sequences_list = self.train_batch_set[1][seq_idx]
        train_sequences_dy_list = self.train_batch_set[0][seq_idx]  # [B L]
        train_uid_track = self.train_batch_set[2][seq_idx]  # [B L]
        labels = self.target_batch_set[seq_idx]
        # generate group attributes padding
        for i in train_sequences_list:
            item_attribute = [0] * self.args.attributes_num
            for att in self.item_att_dict[str(i)]:
                item_attribute[int(att)] = 1
            item_attributes.append(item_attribute)
        cur_tensors = (
            torch.tensor(seq_idx, dtype=torch.long),
            torch.tensor(train_sequences_list, dtype=torch.long),
            torch.tensor(train_sequences_dy_list, dtype=torch.long),
            torch.tensor(train_uid_track, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(item_attributes, dtype=torch.long)
        )
        return cur_tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_dir = "./data"
    args.dataset_name = "amazon_Video_games"
