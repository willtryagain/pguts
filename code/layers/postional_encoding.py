import math
from typing import Optional

import torch
from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP


class PositionalEncoding(nn.Module):
    """
    Implementation of the positional encoding from Vaswani et al. 2017
    """

    def __init__(
        self, d_model, dropout=0.0, max_len=5000, affinity=False, batch_first=True
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if affinity:
            self.affinity = nn.Linear(d_model, d_model)
        else:
            self.affinity = None
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        self.batch_first = batch_first

    def forward(self, x):
        if self.affinity is not None:
            x = self.affinity(x)
        pe = self.pe[: x.size(1), :] if self.batch_first else self.pe[: x.size(0), :]
        x = x + pe
        return self.dropout(x)


class PositionalEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers: int = 1,
        n_nodes: Optional[int] = None,
    ):
        super(PositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(
            out_channels,
            out_channels,
            out_channels,
            n_layers=n_layers,
            activation="relu",
        )
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = StaticGraphEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter("node_emb", None)

    def forward(self, x, node_emb=None, node_index=None):
        if node_emb is None:
            node_emb = self.node_emb(token_index=node_index)
        # x: [b s c], node_emb: [n c] -> [b s n c]
        x = self.lin(x)
        x = self.activation(x.unsqueeze(-2) + node_emb)
        out = self.mlp(x)
        out = self.positional(out)
        return out
