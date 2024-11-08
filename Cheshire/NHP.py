import torch
import torch.nn as nn
import torch_geometric.nn as gnn
# from torch_scatter import scatter
from utils import create_hyperedge_index


class NHP(nn.Module):
    def __init__(self, input_dim, emb_dim, conv_dim):
        super(NHP, self).__init__()
        self.linear_encoder = nn.Linear(input_dim[1], emb_dim)
        self.graph_conv = gnn.GraphConv(emb_dim, conv_dim)
        self.relu = nn.ReLU()
        self.max_pool = gnn.global_max_pool
        self.linear = nn.Linear(conv_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature, incidence_matrix):
        x = self.linear_encoder(feature)
        x, hyperedge_index = self.partition(x, incidence_matrix)
        edge_index, batch = self.expansion(hyperedge_index)
        x = self.relu(self.graph_conv(x, edge_index))
        x_max = self.max_pool(x, batch)
        x_min = self.min_pool(x, batch)
        return self.sigmoid(self.linear(x_max - x_min))

    @staticmethod
    # def min_pool(x, batch):
    #     size = int(batch.max().item() + 1)
    #     return scatter(x, batch, dim=0, dim_size=size, reduce='min')
    def min_pool(x, batch):
        size = int(batch.max().item() + 1)
        result = []
        for i in range(size):
            batch_mask = (batch == i)
            result.append(x[batch_mask].min(dim=0)[0])
        return torch.stack(result)

    @staticmethod
    def expansion(hyperedge_index):
        node_set = hyperedge_index[0]
        b = hyperedge_index[1].int()
        edge_index = torch.empty((2, 0), dtype=torch.int64, device=node_set.device)
        batch = torch.empty(len(node_set), dtype=torch.int64, device=node_set.device)
        for i in range(b[-1] + 1):
            nodes = node_set[b == i]
            batch[nodes.long()] = i
            num_nodes = len(nodes)
            adj_matrix = torch.ones(num_nodes, num_nodes, device=node_set.device) - torch.eye(num_nodes, device=node_set.device)
            row, col = torch.where(adj_matrix)
            row, col = nodes[row], nodes[col]
            edge = torch.cat((row.view(1, -1), col.view(1, -1)), 0)
            edge_index = torch.cat((edge_index, edge), dim=1)
        return edge_index, batch

    @staticmethod
    def partition(x, incidence_matrix):
        hyperedge_index = create_hyperedge_index(incidence_matrix)
        node_set, sort_index = torch.sort(hyperedge_index[0])
        hyperedge_index[1] = hyperedge_index[1][sort_index]
        x = x[node_set.long(), :]
        hyperedge_index[0] = torch.arange(0, len(hyperedge_index[0]))
        index_set, sort_index = torch.sort(hyperedge_index[1])
        hyperedge_index[1] = index_set
        hyperedge_index[0] = hyperedge_index[0][sort_index]
        return x, hyperedge_index
