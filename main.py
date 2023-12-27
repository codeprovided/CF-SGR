import argparse
import time
import torch
import json
import numpy as np
import pickle
import os
from Dataset import PretrainDataset, TestPreTrainDataset, TrainUserDataset, TrainGroupDataset, data_partition, \
    EvalUserDataset
from utils import hypergraph_utils, common_utils
from evaluate import model_evaluate
from model import SIFSGR
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/", type=str)
parser.add_argument("--save_dir", default="./result", type=str)
parser.add_argument("--save_file", default="/model_params.pt", type=str)
parser.add_argument("--save_model_file", default="/model_final.pt", type=str)
parser.add_argument("--dataset_name", default="Toys_and_Games", type=str)
parser.add_argument("--test_res_log_file", default="./result", help='result log file store dictionary')

# model args
parser.add_argument("--model_name", default="SIF-SGR", type=str)
parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of the transformer model")
parser.add_argument("--dropout_ratio", default=0.1, type=float)
parser.add_argument("--max_sequence_length", default=50, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--train_u_epochs", default=2, type=int)
parser.add_argument("--train_g_epochs", default=2, type=int)
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--gcn_layers", default=3, type=int)
parser.add_argument("--pre_epochs", default=2, type=int)
parser.add_argument("--eval_freq", default=1, type=int, help="how many epochs evaluate the model")
parser.add_argument("--seed", default=9999, type=int)
parser.add_argument("--embedding_size", default=64, type=int)
parser.add_argument("--target_length", default=1, type=int)
parser.add_argument("--rec_weight", default=0.8, type=float)
parser.add_argument("--cl_weight", default=0.2, type=float)
parser.add_argument("--i_neg_sample_size", default=99, type=int, help="item negative sample per item")

args = parser.parse_args()
common_utils.set_seed(args.seed)
common_utils.check_save_path(args.save_dir)

if torch.cuda.is_available():
    args.cuda = True
    device = torch.device("cuda")
device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
# print(device)

# data preprocessing
user_res, group_res = data_partition(args)
args.user_num = user_res[-1]
args.item_num = user_res[-2]
args.group_num = group_res[-1]
args.item_g_num = group_res[-2]
args.group_train_time = group_res[2][2]
users_list = [u for u in range(1, args.user_num)]
items_list = [i for i in range(1, args.item_num)]
groups_list = [g for g in range(1, args.group_num)]
g_items_list = [i for i in range(1, args.item_g_num)]
assert args.item_num > args.item_g_num, 'the user-item number should be larger than the group-item number'
# reorder the user-item interaction data and generate the mapping dicts
subgraph_mapping_i, subgraph_mapping_u, subgraph_G = \
    hypergraph_utils.generate_subgraph_mapping(user_res[0], user_res[2][0], user_res[2][-2], gu_flag=0)
# reorder the group-item interaction data and generate the mapping dicts
subgraph_mapping_g_i, subgraph_mapping_g, subgraph_g_G = \
    hypergraph_utils.generate_subgraph_mapping(group_res[0], group_res[2][0], group_res[2][-2], gu_flag=1)
dense_matg = subgraph_G[1]['G'].todense()
dense_mate = subgraph_G[1]['E'].todense()
# generate the dynamic item/user mapping dicts
subgraph_mapping_i, reversed_subgraph_mapping_i, subgraph_sequence_i, reversed_subgraph_mapping_latest_i, time_list, item_dy_num = \
    hypergraph_utils.generate_subgraph_mapping_dynamic(subgraph_mapping_i, args.item_num)
subgraph_mapping_u, reversed_subgraph_mapping_u, subgraph_sequence_u, reversed_subgraph_mapping_latest_u, time_list_u, user_dy_num = \
    hypergraph_utils.generate_subgraph_mapping_dynamic(subgraph_mapping_u, args.user_num)
# generate the dynamic item/group mapping dicts
subgraph_mapping_g_i, reversed_subgraph_mapping_g_i, subgraph_sequence_g_i, reversed_subgraph_mapping_latest_g_i, \
time_list_g_i, g_item_dy_num = hypergraph_utils.generate_subgraph_mapping_dynamic(subgraph_mapping_g_i, args.item_g_num)
subgraph_mapping_g, reversed_subgraph_mapping_g, subgraph_sequence_g, reversed_subgraph_mapping_latest_g, time_list_g, group_dy_num = \
    hypergraph_utils.generate_subgraph_mapping_dynamic(subgraph_mapping_g, args.group_num)
dy_num_set = (user_dy_num, item_dy_num, group_dy_num, g_item_dy_num)
args.user_dy_num, args.item_dy_num, args.group_dy_num, args.g_item_dy_num = user_dy_num, item_dy_num, group_dy_num, g_item_dy_num
ui_neg_dy = np.zeros((args.user_num, args.i_neg_sample_size + 1), dtype=int)
gi_neg_dy = np.zeros((args.group_num, args.i_neg_sample_size + 1), dtype=int)
ui_neg_test = user_res[-3]
gi_neg_test = group_res[-3]
assert len(ui_neg_test.keys()) == args.user_num
for u in range(1, args.user_num):
    for i in range(args.i_neg_sample_size + 1):
        ui_neg_dy[u][i] = subgraph_sequence_i[ui_neg_test[u][i]][-1]
for g in range(1, args.group_num):
    for i in range(args.i_neg_sample_size + 1):
        gi_neg_dy[g][i] = subgraph_sequence_g_i[gi_neg_test[g][i]][-1]

# load side information from item_attributes_dict
item_attr_path = f"{os.path.join(args.data_dir, args.dataset_name)}/{args.dataset_name}_item2attributes.json"
item_attributes_dict = json.loads(open(item_attr_path).readline())
attributes_num = 0
for value in item_attributes_dict.values():
    if value:
        attributes_num = max(attributes_num, max(value))
args.attributes_num = attributes_num + 1
item_attributes_dict['0'] = []
ori_num_set = (args.user_num, args.item_num, args.group_num, args.item_g_num)
dy_num_set = (user_dy_num, item_dy_num, group_dy_num, g_item_dy_num)
dataset_dir = os.path.join(args.data_dir, args.dataset_name)
args.gu_dict = common_utils.load_gu_dict(dataset_dir + f"/{args.dataset_name}_gu.csv")
# define pre-train dataset
pre_train_dataset = PretrainDataset(subgraph_mapping_i, subgraph_mapping_u, subgraph_sequence_i, subgraph_sequence_u,
                                    users_list, items_list, user_res[0], user_res[2][0], args)
test_pre_train_dataset = TestPreTrainDataset(pre_train_dataset.pre_test_batch_set, ui_neg_test)
# define train/valid/test user datasets
train_user_dataset = TrainUserDataset(subgraph_mapping_i, subgraph_mapping_u, subgraph_sequence_i, subgraph_sequence_u,
                                      users_list, user_res[0], user_res[2][0], item_attributes_dict, args)
eval_user_dataset = EvalUserDataset(users_list, train_user_dataset.test_batch_set, ui_neg_test, ui_neg_dy)
# define train/valid/test group datasets
train_group_dataset = TrainGroupDataset(subgraph_mapping_g_i, subgraph_mapping_g, subgraph_sequence_g_i,
                                        subgraph_sequence_g, groups_list, group_res[0],
                                        group_res[2][0], item_attributes_dict, args)
eval_group_dataset = EvalUserDataset(groups_list, train_group_dataset.test_batch_set, gi_neg_test, gi_neg_dy)
args.train_groups = train_group_dataset.train_batch_set[-1]
params = {'batch_size': args.batch_size, 'shuffle': False}
# 'num_worker'/ 'pin_memory' cannot use in windows
# use user-item data for pre-train
pre_train_dataloader = DataLoader(pre_train_dataset, **params)
test_pre_train_dataloder = DataLoader(test_pre_train_dataset, **params)
user_train_dataloader = DataLoader(train_user_dataset, **params)
group_train_dataloader = DataLoader(train_group_dataset, **params)
user_eval_dataloader = DataLoader(eval_user_dataset, **params)
group_eval_dataloader = DataLoader(eval_group_dataset, **params)

torch.autograd.set_detect_anomaly(True)

model = SIFSGR(args, ori_num_set, dy_num_set, subgraph_G, subgraph_g_G, reversed_subgraph_mapping_i,
               reversed_subgraph_mapping_latest_i, reversed_subgraph_mapping_g_i, subgraph_sequence_u,
               reversed_subgraph_mapping_latest_g_i, time_list, time_list_g_i, args.gcn_layers).to(device)
try:
    # pre-train on user-item interactions
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_NDCG20 = -np.inf
    for epoch in range(args.pre_epochs):
        epoch_st_time = time.time()
        model.train()
        losses = []
        pretrain_data_iter = enumerate(pre_train_dataloader)
        for batch_idx, cur_tensors in pretrain_data_iter:
            cur_tensors = tuple(t.to(device) for t in cur_tensors)
            user_predicts, labels = model(cur_tensors, None, type_m='pretrain')
            user_predicts = user_predicts.view(-1, user_predicts.shape[2])
            labels = labels.view(-1)  # [B*L]
            pretrain_loss = model.rec_loss(user_predicts, labels)
            model.zero_grad()
            losses.append(float(pretrain_loss))
            with torch.autograd.detect_anomaly():
                pretrain_loss.backward(retain_graph=True)
            optimizer_pretrain.step()
        epoch_ed_time = time.time()
        print(
            "epoch: {}, pretrain_time: {}, loss: {}".format(epoch + 1, epoch_ed_time - epoch_st_time, np.mean(losses)))
        if epoch % args.eval_freq == 0:
            HT, NDCG = model_evaluate(model, args, test_pre_train_dataloder, None, device, 'pretrain')
            if NDCG[-2] > best_NDCG20:
                best_NDCG20 = NDCG[-2]
                print(f"update the best pretrain model on epoch: {epoch}, the NDCG20 is {NDCG[-2]}")
                torch.save(model.state_dict(), args.save_dir + args.save_file)

    # user-level hypergraph building
    print("execute user-level hypergraph model training!")
    print("load the best model from pretrain module")
    best_user_NDCG = -np.inf
    model.load_state_dict(torch.load(args.save_dir + args.save_file))
    model = model.to(device)
    # use pretrain predictor for initialing the user and group predictor weight
    model.user_predictor.weight.data = model.pretrain_predictor.weight.data
    model.group_predictor.weight.data = model.pretrain_predictor.weight.data
    optimizer_u = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(args.train_u_epochs):
        model.train()
        epoch_st_time = time.time()
        user_rec_losses = []
        user_cl_losses = []
        joint_loss = 0
        user_data_iter = enumerate(user_train_dataloader)
        user_ori_embs, item_ori_embs = model.user_ori_embedding.weight, model.item_ori_embedding.weight
        all_user_dy_embs, all_item_dy_embs = model._build_ul_hypergraph(user_ori_embs, item_ori_embs, model.args.device)
        embedding_set = (all_user_dy_embs, all_item_dy_embs)
        for batch_idx, cur_tensors in user_data_iter:
            cur_tensors = tuple(t.to(device) for t in cur_tensors)
            item_attributes = cur_tensors[-1]
            masked_item_sequence = (cur_tensors[1] != 0).float()  # [B L]
            user_preferences, user_predicts, labels = model(cur_tensors[:-1], embedding_set, type_m='user')
            user_predicts = user_predicts.view(-1, user_predicts.shape[2])
            labels = labels.view(-1)  # [B*L]
            user_rec_loss = model.rec_loss(user_predicts, labels)
            # user-level contrastive learning
            cl_loss = model.contextual_contrastive_learning(user_preferences, item_attributes, masked_item_sequence)
            joint_loss = user_rec_loss * model.args.rec_weight + cl_loss * model.args.cl_weight
            user_rec_losses.append(float(user_rec_loss.item() * model.args.rec_weight))
            user_cl_losses.append(float(cl_loss.item() * model.args.cl_weight))
            model.zero_grad()
            with torch.autograd.detect_anomaly():
                joint_loss.backward(retain_graph=True)
            optimizer_u.step()
        epoch_ed_time = time.time()
        print("epoch: {}, train_time: {}, rec_loss: {}, cl_loss: {}".format(epoch + 1, epoch_ed_time - epoch_st_time, \
                                                                            np.mean(user_rec_losses),
                                                                            np.mean(user_cl_losses)))
        if epoch % args.eval_freq == 0:
            model.eval()
            HT, NDCG = model_evaluate(model, args, user_eval_dataloader, embedding_set, device, 'user')
            if best_user_NDCG < NDCG[-2]:
                best_user_NDCG = NDCG[-2]
                print("update the best user training epoch at:", epoch)

    # group level hypergraph building
    print("execute group-level hypergraph model training!")
    optimizer_g = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_group_NDCG = -np.inf
    for epoch in range(args.train_g_epochs):
        model.train()
        epoch_st_time = time.time()
        group_rec_losses = []
        group_cl_losses = []
        group_joint_loss = 0
        group_data_iter = enumerate(group_train_dataloader)
        all_u_embs, all_ui_embs = model.user_ori_embedding.weight, model.item_ori_embedding.weight
        all_g_embs, all_gi_embs = model.group_ori_embedding.weight, model.g_item_ori_embedding.weight
        all_user_dy_embs, _ = model._build_ul_hypergraph(all_u_embs, all_ui_embs, model.args.device)
        item_dy_emb_list, group_dy_emb_list = model._build_gl_hypergraph(all_g_embs, all_gi_embs, model.args.device)
        embedding_set = (all_user_dy_embs, item_dy_emb_list, group_dy_emb_list, all_gi_embs)
        for batch_idx, cur_tensors in group_data_iter:
            cur_tensors = tuple(t.to(device) for t in cur_tensors)
            item_attributes = cur_tensors[-1]
            masked_item_sequence = (cur_tensors[1] != 0).float()  # [B L]
            group_preferences, group_predicts, labels = model(cur_tensors[:-1], embedding_set, type_m='group')
            group_predicts = group_predicts.view(-1, group_predicts.shape[2])
            labels = labels.view(-1)  # [B*L]
            group_rec_loss = model.rec_loss(group_predicts, labels)
            # group-level contrastive learning
            g_cl_loss = model.contextual_contrastive_learning(group_preferences, item_attributes, masked_item_sequence)
            group_joint_loss = group_rec_loss * args.rec_weight + g_cl_loss * args.cl_weight
            group_rec_losses.append(float(group_rec_loss.item() * args.rec_weight))
            group_cl_losses.append(float(g_cl_loss.item() * args.cl_weight))
            model.zero_grad()
            with torch.autograd.detect_anomaly():
                group_joint_loss.backward(retain_graph=True)
            optimizer_g.step()
        epoch_ed_time = time.time()
        print(
            "epoch: {}, train_time: {}, rec_loss: {}, cl_loss: {}".format(epoch + 1, epoch_ed_time - epoch_st_time, \
                                                                          np.mean(group_rec_losses),
                                                                          np.mean(group_cl_losses)))
        if epoch % args.eval_freq == 0:
            HT, NDCG = model_evaluate(model, args, group_eval_dataloader, embedding_set, device, 'group')
            if NDCG[-2] > best_group_NDCG:
                best_group_NDCG = NDCG[-2]
                torch.save(model, args.save_dir + args.save_model_file)
except KeyboardInterrupt as e:
    print('*' * 90)
    print("user exiting from training early!")

model = torch.load(args.save_dir + args.save_model_file, map_location='cuda')
model = model.to(device)
model.eval()

# evaluate the model on user-level and group-level
user_ori_embs, item_ori_embs = model.user_ori_embedding.weight, model.item_ori_embedding.weight
all_user_dy_embs, all_item_dy_embs = model._build_ul_hypergraph(user_ori_embs, item_ori_embs, model.args.device)
embedding_set = (all_user_dy_embs, all_item_dy_embs)
HT, NDCG = model_evaluate(model, args, user_eval_dataloader, embedding_set, device, 'user')

all_u_embs, all_ui_embs = model.user_ori_embedding.weight, model.item_ori_embedding.weight
all_g_embs, all_gi_embs = model.group_ori_embedding.weight, model.g_item_ori_embedding.weight
all_user_dy_embs, _ = model._build_ul_hypergraph(all_u_embs, all_ui_embs, model.args.device)
item_dy_emb_list, group_dy_emb_list = model._build_gl_hypergraph(all_g_embs, all_gi_embs, model.args.device)
embedding_set = (all_user_dy_embs, item_dy_emb_list, group_dy_emb_list, all_gi_embs)
HT, NDCG = model_evaluate(model, args, group_eval_dataloader, embedding_set, device, 'group')

