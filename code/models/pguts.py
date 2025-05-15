from argparse import ArgumentParser as ArgParser

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import OptTensor
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers.norm import LayerNorm

from ..layers import PGUTS_Branch, PositionalEncoder


class PGUTS_Layer(nn.Module):
    def __init__(
        self,
        window,
        n_nodes,
        hidden_size,
        u_size,
        n_heads,
        ff_size,
        use_pg_enc,
        ts_order,
        factor_t,
    ):
        super().__init__()
        self.factor_t = factor_t
        self.encoder = nn.ModuleList()
        self.missing_emb, self.present_emb = nn.ModuleList(), nn.ModuleList()
        in_size = 0
        for l in range(len(factor_t)):
            in_size += window // factor_t[l]
            encoder = PGUTS_Branch(
                hidden_size,
                u_size,
                0,
                1,
                1,
                factor_t[l],
                n_heads,
                ff_size,
                use_pg_enc,
                ts_order,
                n_nodes=n_nodes,
            )
            self.missing_emb.append(StaticGraphEmbedding(n_nodes, hidden_size))
            self.present_emb.append(StaticGraphEmbedding(n_nodes, hidden_size))
            self.encoder.append(encoder)
        self.mlp = MLP(len(factor_t) * hidden_size, hidden_size, n_layers=2)

    def forward(self, h, mask, node_index):
        out = []
        for l in range(len(self.factor_t)):
            h_l = h.clone()
            h_l = torch.where(
                mask.bool(),
                h_l + self.present_emb[l](token_index=node_index),
                h_l + self.missing_emb[l](token_index=node_index),
            )
            h_l = self.encoder[l](h_l)
            out.append(h_l)
        out = torch.concat(out, dim=-1)
        out = self.mlp(out)
        return out


class PGUTS(nn.Module):
    def __init__(
        self,
        window,
        n_layers,
        n_nodes,
        output_size=1,
        hidden_size=64,
        u_size=4,
        factor_t=[1, 2],
        n_heads=4,
        ff_size=128,
        use_pg_enc=False,
        ts_order=True,
    ):
        super().__init__()
        self.window = window
        self.n_layers = n_layers
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        self.h_enc = MLP(1, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)
        self.merge_norm = LayerNorm(hidden_size)
        self.u_enc = PositionalEncoder(
            in_channels=u_size, out_channels=hidden_size, n_layers=2, n_nodes=n_nodes
        )
        self.factor_t = factor_t
        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(1, hidden_size)
            encoder = PGUTS_Layer(
                window,
                n_nodes,
                hidden_size,
                u_size,
                n_heads,
                ff_size,
                use_pg_enc,
                ts_order,
                factor_t,
            )
            readout = MLP(hidden_size, hidden_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(
        self,
        x,
        u: Tensor,
        mask: Tensor,
        node_index: OptTensor = None,
        target_nodes: OptTensor = None,
    ):
        x = x * mask
        q = self.u_enc(u, node_index=node_index)
        h = self.h_enc(x) + q
        h = torch.where(mask.bool(), h, q)
        h = self.h_norm(h)

        imputations = []

        for l in range(self.n_layers):
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            h = self.encoder[l](h, mask, node_index)
            # Read from H to get imputations
            target_readout = self.readout[l](h)
            imputations.append(target_readout)
        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list("--factor_t", type=int, nargs="*", default=[])
        return parser
