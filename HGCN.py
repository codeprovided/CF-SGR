import torch
import torch.nn as nn
from math import sqrt
from torch.nn.parameter import Parameter


def _init_weight(k, input_dim, output_dim, device):
    weight = {}
    bias = {}
    for i in range(k):
        weight['weight_%d' % i] = Parameter(torch.Tensor(input_dim, output_dim)).to(device)
        bias['bias_%d' % i] = Parameter(torch.Tensor(output_dim)).to(device)
        # try another parameters init method: stdv
        nn.init.xavier_uniform_(weight['weight_%d' % i])
        nn.init.zeros_(bias['bias_%d' % i])
    return weight, bias

class HGNN_conv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_ratio, hyper_layer, device):
        super(HGNN_conv, self).__init__()
        self.hyper_layer = hyper_layer
        self.dropout_ratio = dropout_ratio
        # init weight bias
        self.weight, self.bias = _init_weight(hyper_layer, input_dim, output_dim, device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)


    def __call__(self, input_x, adj_matrix_G):
        for i in range(self.hyper_layer):
            if i == 0:
                input_x = input_x.matmul(self.weight['weight_%d' % i])
                input_x = input_x + self.bias['bias_%d' % i]
                nodes = torch.sparse.mm(adj_matrix_G['G'], input_x)
                nodes = nn.functional.relu(nodes)
            else:
                nodes = nn.functional.dropout(nodes, self.dropout_ratio)
                nodes = nodes.matmul(self.weight['weight_%d' % i])
                nodes = nodes + self.bias['bias_%d' % i]
                nodes = torch.sparse.mm(adj_matrix_G['G'], nodes)
                nodes = nn.functional.relu(nodes)
        temp_nodes = nn.functional.dropout(nodes, self.dropout_ratio)
        edges = torch.sparse.mm(adj_matrix_G['E'], temp_nodes)
        edges = nn.functional.relu(edges)
        return nodes, edges
