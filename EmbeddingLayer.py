import torch
import torch.nn as nn


def init_weight(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class AttentionLayer(nn.Module):
    def __init__(self, embedding_size, drouout_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, 16),
            nn.ReLU(),
            nn.Dropout(drouout_ratio),
            nn.Linear(16, 1)
        )
        init_weight(self.linear[0])
        init_weight(self.linear[-1])

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight


class LinearActivateLayer(nn.Module):
    def __init__(self, embedding_size, dropout_ratio=0):
        super(LinearActivateLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(embedding_size, 1),
            nn.Softmax(0),
            nn.Dropout(dropout_ratio)
        )
        init_weight(self.linear[0])
        init_weight(self.linear[-1])

    def forward(self, x):
        output = self.linear(x)
        return output


class PredictLayer(nn.Module):
    def __init__(self, embedding_size, dropout_ratio):
        super(PredictLayer, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(embedding_size * 3, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, u_emb, i_emb):
        element_emb = torch.mul(u_emb, i_emb)
        new_emb = torch.cat((element_emb, u_emb, i_emb), dim=1)
        outputs = self.predict(new_emb)
        return outputs


class Embedding_Fusing_Bi(nn.Module):
    def __init__(self, embedding_size, dropout_ratio):
        super(Embedding_Fusing_Bi, self).__init__()
        self.LinearActivateLayer = LinearActivateLayer(embedding_size, dropout_ratio)

    def forward(self, user_embedding, item_embedding):
        stack_feature = torch.stack([user_embedding, item_embedding], dim=0)  # [2 B L H]
        feature_score = self.LinearActivateLayer(stack_feature)
        transformed_outputs = torch.sum(feature_score * stack_feature, dim=0)  # [B L H]
        return transformed_outputs


class Embedding_Fusing(nn.Module):
    def __init__(self, embedding_size, dropout_ratio):
        super(Embedding_Fusing, self).__init__()
        self.LinearActivateLayer = LinearActivateLayer(embedding_size, dropout_ratio)

    def forward(self, user_dy_embedding, item_dy_embedding, itm_ori_embedding):
        stack_feature = torch.stack([user_dy_embedding, item_dy_embedding, itm_ori_embedding], dim=0)  # [3 B L H]
        feature_score = self.LinearActivateLayer(stack_feature) # [3 B L 1]
        transformed_outputs = torch.sum(feature_score * stack_feature, dim=0)  # [B L H]
        return transformed_outputs
