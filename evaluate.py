import torch
import pandas as pd
import numpy as np
from utils import common_utils


def model_evaluate(model, args, eval_dataloder, embedding_set, device, mode='pretrain'):
    """
    :param model:
    :param args:
    :param eval_dataloder: dataloader
    :param embeddings_set: ori_u/ori_i emb for pretrain, dy_u/dy_i for user, dy_g/dy_i for group
    :param mode: pretrain/user/group
    :param device: cuda/cpu
    :return: HT, NDCG
    """
    model.eval()
    if mode == 'pretrain' or mode == 'user':
        iter_list = np.arange(1, args.user_num)
    else:
        iter_list = np.arange(1, args.group_num)
    pred_list = None
    with torch.no_grad():
        # stop calculating the gradient and constructing the calculation map
        for batch_idx, cur_tensors in enumerate(eval_dataloder):
            cur_tensors = tuple(t.to(device) for t in cur_tensors)
            if mode == 'pretrain':
                user_predicts, labels = model(cur_tensors, None, type_m='pretrain')
                user_scores = user_predicts[:, -1, :]  # [B I]
                user_scores = user_scores.gather(1, labels)  # [B 100]
                # each row in batch_logits represent one user's predicted ranking result
                user_scores = user_scores.cpu().data.numpy().copy()
                if batch_idx == 0:
                    pred_list = user_scores
                else:
                    pred_list = np.append(pred_list, user_scores, axis=0)
            elif mode == 'user':
                # user_ori_embs, item_ori_embs = model.user_ori_embedding.weight, model.item_ori_embedding.weight
                # all_user_dy_embs, all_item_dy_embs = model._build_ul_hypergraph(user_ori_embs, item_ori_embs, device)
                user_preferences, user_predicts, labels = model(cur_tensors, embedding_set, type_m='user')
                user_scores = user_predicts[:, -1, :]  # [B I]
                user_scores = user_scores.gather(1, labels)  # [B 100]
                user_scores = user_scores.cpu().data.numpy().copy()
                if batch_idx == 0:
                    pred_list = user_scores
                else:
                    pred_list = np.append(pred_list, user_scores, axis=0)
            else:
                group_preferences, group_predicts, labels = model(cur_tensors, embedding_set, type_m='group')
                group_scores = group_predicts[:, -1, :]  # [B I]
                group_scores = group_scores.gather(1, labels)  # [B 100]
                group_scores = group_scores.cpu().data.numpy().copy()
                if batch_idx == 0:
                    pred_list = group_scores
                else:
                    pred_list = np.append(pred_list, group_scores, axis=0)
        # calculate evaluate matrics
        k_list = [5, 10, 20, 50]
        HT = [0.0 for k in k_list]
        NDCG = [0.0 for k in k_list]
        HT, NDCG = common_utils.calculate_evaluate_metric(args, pred_list, k_list, HT, NDCG, iter_list, mode)
    return HT, NDCG


