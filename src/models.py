"""Model: MLP baseline dan GNN graf-individual (GCN / GATv2) dengan beberapa pilihan readout."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation, GATv2Conv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import dropout_edge


class MLP(nn.Module):
    """Baseline non-graf: input = vektor segitiga atas FC (batch.x berukuran B x D)."""

    def __init__(self, in_dim, hidden=128, dropout=0.5, n_kelas=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, n_kelas))

    def forward(self, batch):
        return self.net(batch.x)


class GNNGraf(nn.Module):
    """GNN untuk satu graf otak per subjek.

    conv     : "gatv2" atau "gcn"
    readout  : "mean", "max", "meanmax", atau "attn" (attentional aggregation)
    """

    def __init__(self, in_dim, hidden=64, n_layer=2, heads=2, conv="gatv2", readout="meanmax",
                 dropout=0.5, dropout_input=0.2, dropedge=0.2, n_kelas=2):
        super().__init__()
        self.conv_tipe, self.readout, self.dropout = conv, readout, dropout
        self.dropout_input, self.dropedge = dropout_input, dropedge
        self.proj = nn.Linear(in_dim, hidden)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layer):
            if conv == "gatv2":
                self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1, dropout=dropout))
            else:
                self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        if readout == "attn":
            self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(),
                                                             nn.Linear(hidden // 2, 1)))
        d_graf = hidden * 2 if readout == "meanmax" else hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_graf, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_kelas))

    def embed(self, batch):
        x, ei, ea, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        if self.training and self.dropedge > 0:
            ei, m = dropout_edge(ei, p=self.dropedge)
            ea = ea[m]
        x = F.dropout(x, self.dropout_input, self.training)
        h = F.elu(self.proj(x))
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_tipe == "gatv2":
                h2 = conv(h, ei, ea)
            else:
                h2 = conv(h, ei, edge_weight=ea.view(-1).abs())
            h = F.elu(norm(h + h2))
            h = F.dropout(h, self.dropout, self.training)
        if self.readout == "mean":
            return global_mean_pool(h, b)
        if self.readout == "max":
            return global_max_pool(h, b)
        if self.readout == "attn":
            return self.pool(h, b)
        return torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], dim=1)

    def forward(self, batch):
        return self.head(self.embed(batch))
