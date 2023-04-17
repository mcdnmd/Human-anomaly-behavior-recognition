import torch
from torch_geometric.nn import GCNConv
from torch.nn.functional import dropout


class GCN(torch.nn.Module):
    def __init__(self, input_feature: int, hidden_channels: int, out_classes: int):
        super().__init__()
        self.conv1 = GCNConv(input_feature, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = dropout(x, p=0.5, traning=self.traning)
        x = self.conv2(x, edge_index)
        return x
