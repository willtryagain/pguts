import numpy as np
import scipy.sparse as sp
import torch
from einops import rearrange
from torch import nn
from tsl.nn.blocks.encoders import MLP
from tsl.nn.blocks.encoders.transformer import TransformerLayer
from tsl.nn.layers import PositionalEncoding
from tsl.nn.layers.norm import LayerNorm


def to_sparse_adj(dense_adj):
    if dense_adj.device.type == "cuda":
        dense_adj = dense_adj.cpu()
    sparse_mat = sp.coo_matrix(dense_adj)
    u = sparse_mat.row
    v = sparse_mat.col
    edge_index = torch.from_numpy(np.vstack([u, v])).cuda().long()
    edge_weight = torch.from_numpy(sparse_mat.data).cuda()
    return edge_index, edge_weight


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DictWithAttributes(dict):
    def __getattr__(self, key):
        # Check if the key exists in the dictionary
        if key in self:
            return self[key]
        else:
            # Raise an AttributeError if the key doesn't exist
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        # Set the attribute as a key-value pair in the dictionary
        self[key] = value


class gPool(nn.Module):
    def __init__(self, dim=1, hidden_size=64, factor=4, ff_size=128, no_pe=True):
        super().__init__()
        self.no_pe = no_pe
        self.hidden_size = hidden_size
        self.factor = factor
        self.p = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.dim = dim
        self.norm = LayerNorm(hidden_size)
        if dim == 2:
            self.pe = PositionalEncoding(hidden_size)
            self.enc = TransformerLayer(
                input_size=hidden_size,
                axis="nodes",
                hidden_size=hidden_size,
                ff_size=ff_size,
                n_heads=4,
                causal=False,
            )
        else:
            self.pe = PositionalEncoding(hidden_size)
            self.enc = TransformerLayer(
                input_size=hidden_size,
                hidden_size=hidden_size,
                ff_size=ff_size,
                n_heads=4,
                causal=False,
            )

    def most_frequent_indices(self, idx, k):
        idx = idx.flatten().cpu().numpy()
        freq = {}
        for i in idx:
            if i not in freq:
                freq[i] = 0
            freq[i] += 1
        vals = [(cnt, i) for i, cnt in freq.items()]
        vals.sort(key=lambda pair: -pair[0])
        vals = vals[:k]
        return [v[1] for v in vals]

    def forward(self, x):
        y = (x @ self.p) / torch.linalg.norm(self.p)  # b, s, n
        idx = torch.argsort(y, dim=self.dim, descending=True)
        k = int(y.shape[self.dim] // self.factor)
        idx = self.most_frequent_indices(idx, k)
        idx = torch.tensor(idx).cuda()
        num_nodes = x.shape[1]
        if self.dim == 1:
            y = y[:, idx]
            x = x[:, idx]
        else:
            y = y[:, :, idx]
            x = x[:, :, idx]

        y = torch.sigmoid(y)
        x = x * y.unsqueeze(-1)
        x = self.norm(x)  # O_i
        if self.dim == 1:
            if not self.no_pe:
                x = self.pe(x)
            x = self.enc(x)  # P_u
        else:
            if not self.no_pe:
                x = x.transpose(1, 2)
                x = self.pe(x)
                x = x.transpose(1, 2)
            x = self.enc(x)
        return x, num_nodes, idx


def P(x, pooling_mods, skip_te=False):
    pool_data_list = []
    for i in range(len(pooling_mods)):
        data = DictWithAttributes()
        data["x"] = x.clone()  # store the input to the pool
        if skip_te:
            num_nodes = x.shape[1]
            idx = torch.arange(num_nodes)
        else:
            x, num_nodes, idx = pooling_mods[i](x)
        data["num_nodes"] = num_nodes
        data["idx"] = idx
        pool_data_list.append(data)
    return x, pool_data_list


def U(x, dim, pool_data_list, encs, skip_te=False):
    num_iter = len(encs)
    for i in range(num_iter):
        data = pool_data_list[num_iter - 1 - i]  # reverse unpooling
        idx = data.idx
        x_cur = data.x.clone()
        if dim == 1:
            x_cur[:, idx] += x
            if not skip_te:
                x = encs[i](x_cur)
            else:
                x = x_cur

        else:
            x_cur[:, :, idx] += x  # Q_i

            if not skip_te:
                x = encs[i](x_cur)
    return x


class PGUTS_Branch(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        u_size=4,
        space_count=1,
        time_count=1,
        factor_s=1,
        factor_t=6,
        n_heads=4,
        ff_size=128,
        use_pg_enc=True,
        ts_order=True,
        no_unpool=False,
        n_nodes=207,
    ):
        super().__init__()
        self.space_count = space_count
        self.time_count = time_count
        self.pe = PositionalEncoding(hidden_size)
        if n_nodes > 325:
            self.pg_pe = PositionalEncoding(hidden_size, max_len=10488)
        elif n_nodes > 207:
            self.pg_pe = PositionalEncoding(hidden_size, max_len=7800)
        else:
            self.pg_pe = PositionalEncoding(hidden_size)
        self.u_enc = MLP(u_size, hidden_size, n_layers=2)
        self.h_enc = MLP(1, hidden_size, n_layers=2)
        self.use_pg_enc = use_pg_enc
        self.ts_order = ts_order
        self.no_unpool = no_unpool
        if self.space_count:
            self.enc_init_s = TransformerLayer(
                input_size=hidden_size,
                axis="nodes",
                hidden_size=hidden_size,
                ff_size=ff_size,
                n_heads=4,
                causal=False,
            )
            self.SP = nn.ModuleList([gPool(2, hidden_size, factor_s)] * space_count)
            self.SU = nn.ModuleList(
                [
                    TransformerLayer(
                        input_size=hidden_size,
                        axis="nodes",
                        hidden_size=hidden_size,
                        ff_size=ff_size,
                        n_heads=4,
                        causal=False,
                    )
                ]
                * space_count
            )
        if self.time_count:
            self.enc_init_t = TransformerLayer(
                input_size=hidden_size,
                hidden_size=hidden_size,
                ff_size=ff_size,
                n_heads=4,
                causal=False,
            )
            self.TP = nn.ModuleList([gPool(1, hidden_size, factor_t)] * time_count)
            self.TU = nn.ModuleList(
                [
                    TransformerLayer(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        ff_size=ff_size,
                        n_heads=4,
                        causal=False,
                    )
                ]
                * time_count
            )
        self.pg_encoder = TransformerLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_heads=n_heads,
            causal=False,
        )

    def forward(self, h):
        if self.ts_order:
            # time pool  # b, s, n, c -> b, k_t, n, c
            if self.time_count:
                h = self.pe(h)
                h = self.enc_init_t(h)
                h, time_pool_data_list = P(h, self.TP)

            # space pool  b, k_t, n, c - b k_t k_n c
            if self.space_count:
                h = h.transpose(1, 2)  # b, n, s, c
                h = self.pe(h)  # add pos enc
                h = h.transpose(1, 2)  # b, s, n, c
                h = self.enc_init_s(h)
                h, space_pool_data_list = P(h, self.SP)
        else:
            # space pool  b, k_t, n, c - b k_t k_n c
            if self.space_count:
                h = h.transpose(1, 2)  # b, n, s, c
                h = self.pe(h)  # add pos enc
                h = h.transpose(1, 2)  # b, s, n, c
                h = self.enc_init_s(h)
                h, space_pool_data_list = P(h, self.SP)
            # time pool  # b, s, n, c -> b, k_t, n, c
            if self.time_count:
                h = self.pe(h)
                h = self.enc_init_t(h)
                h, time_pool_data_list = P(h, self.TP)
        if self.use_pg_enc:
            # tf
            k_t = h.shape[1]
            h = rearrange(h, "b k_t k_n c -> b (k_t k_n) 1 c")
            h = self.pg_pe(h)
            h = self.pg_encoder(h)
            temp = rearrange(h, "b (k_t k_n) 1 c -> b k_t k_n c", k_t=k_t).clone()
        else:
            temp = h
        if self.no_unpool:
            return temp

        if self.ts_order:
            # space unpool
            if self.space_count:
                temp = U(temp, 2, space_pool_data_list, self.SU)
            # time unpool
            if self.time_count:
                temp = U(temp, 1, time_pool_data_list, self.TU)
        else:
            # time unpool
            if self.time_count:
                temp = U(temp, 1, time_pool_data_list, self.TU)
            # space unpool
            if self.space_count:
                temp = U(temp, 2, space_pool_data_list, self.SU)
        return temp
