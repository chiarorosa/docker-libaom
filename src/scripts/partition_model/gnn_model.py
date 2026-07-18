#!/usr/bin/env python3
"""Approach B (Stage 1): GNN sobre o quadtree do superbloco, com camadas expressivas
do PyTorch Geometric (SAGE/GAT/GIN). Com n_layers=0 NÃO há camada de grafo -> por
construção um MLP independente por nó (o baseline da ablação estrutural). Adota-se
PyG para dar à estrutura o melhor tiro; a escolha da lib não é metodológica -- a
ablação n_layers=0 vs n_layers>=1 é."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GINConv, SAGEConv


def _make_conv(layer, hidden):
    if layer == "sage":
        return SAGEConv(hidden, hidden)
    if layer == "gat":
        return GATConv(hidden, hidden)
    if layer == "gin":
        return GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden)))
    raise ValueError("layer desconhecido: {}".format(layer))


class TreeGNN(nn.Module):
    def __init__(self, in_features=36, hidden=64, layer="sage", n_layers=2,
                 num_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())
        self.convs = nn.ModuleList([_make_conv(layer, hidden)
                                    for _ in range(n_layers)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for conv in self.convs:                 # vazio se n_layers=0 -> MLP puro
            h = torch.relu(conv(h, edge_index))
        return self.head(h)


def build_gnn(in_features=36, hidden=64, layer="sage", n_layers=2):
    return TreeGNN(in_features, hidden, layer, n_layers)
