"""
CSCLog model definition.
Extracted from main.ipynb so it can be imported by train.py and evaluate.py.

Components:
  (a) FTEncoder        – fuse sentence embedding + time delta
  (b) LSTMEncoder      – sequence-level / component-level LSTM
  (c) IREncoder        – inter-component relation GCN
  (d) Attention pooling + classifier
"""

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class MLPLayer(nn.Module):
    def __init__(self, dmodel, hid_size, drop):
        super().__init__()
        self.drop = drop
        self.fc0 = nn.Linear(dmodel, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.fc1(x))
        return x


class FTEncoder(nn.Module):
    """Fuse template embedding + time delta into a fixed-size vector.

    pattern=0  → cat(sen, time) → linear
    pattern=1  → linear(sen) || linear(time)    (proportional split)
    pattern=2  → linear(sen) + linear(time)     (additive)
    """
    def __init__(self, sen_size, hidden_size, alpha=0.5, pattern=0):
        super().__init__()
        self.pattern = pattern
        assert pattern in [0, 1, 2]

        if pattern == 1:
            assert 0 < alpha < 1
            sen_fc_size  = int(hidden_size * alpha)
            time_fc_size = hidden_size - sen_fc_size
            self.sen_fc  = nn.Linear(sen_size, sen_fc_size)
            self.time_fc = nn.Linear(1, time_fc_size)
        elif pattern == 0:
            self.cat_fc  = nn.Linear(sen_size + 1, hidden_size)
        elif pattern == 2:
            self.sen_fc  = nn.Linear(sen_size, hidden_size)
            self.time_fc = nn.Linear(1, hidden_size)

    def forward(self, x):
        sen_x, time_x = x          # sen_x: (B, W, D)  time_x: (B, W)
        if self.pattern == 0:
            cat = torch.cat((sen_x, time_x.unsqueeze(-1)), -1)
            return self.cat_fc(cat)
        elif self.pattern == 1:
            return torch.cat((self.sen_fc(sen_x), self.time_fc(time_x.unsqueeze(-1))), -1)
        else:
            return self.sen_fc(sen_x) + self.time_fc(time_x.unsqueeze(-1))


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]


class IREncoder(nn.Module):
    """Inter-component Relation encoder (GCN with learned edge weights)."""

    def __init__(self, dmodel, mlp_hid_size, gcn_hid_size, drop, com_num):
        super().__init__()
        self.dmodel      = dmodel
        self.drop        = drop
        self.com_num     = com_num

        self.edge_mlp = MLPLayer(2 * dmodel, mlp_hid_size, drop)
        self.mlp_out  = nn.Linear(mlp_hid_size, 1)
        self.GCN0     = GCNConv(dmodel, gcn_hid_size)
        self.GCN1     = GCNConv(gcn_hid_size, dmodel)

    def _build_edge_index(self, node_indices, device):
        src, dst = [], []
        for i in range(len(node_indices)):
            for j in range(i + 1, len(node_indices)):
                src.append(node_indices[i])
                dst.append(node_indices[j])
        return torch.stack([
            torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(dst, dtype=torch.long, device=device),
        ])

    def _gumbel_softmax(self, x, axis=1):
        t = x.transpose(axis, 0).contiguous()
        s = F.softmax(t, dim=0)
        return s.transpose(axis, 0)

    def forward(self, x, index):
        device = x.device
        padding = torch.zeros(self.com_num, self.dmodel, device=device)
        padding[index] = x

        edge_index = self._build_edge_index(index, device)
        edge_x = torch.cat([padding[edge_index[0]], padding[edge_index[1]]], -1)
        edge_x = self.edge_mlp(edge_x)
        edge_x = self.mlp_out(edge_x)
        edge_w = self._gumbel_softmax(edge_x)

        out = F.relu(self.GCN0(padding, edge_index, edge_w))
        out = F.dropout(out, self.drop, training=self.training)
        out = self.GCN1(out, edge_index, edge_w)
        return out[index]


# ─────────────────────────────────────────────────────────────────────────────
# Full CSCLog Model
# ─────────────────────────────────────────────────────────────────────────────

class CSCLog(nn.Module):
    """
    Parameters
    ----------
    input_size  : dimension of the sentence embedding
    com_num     : total number of unique components
    hidden_size : list/tuple [ft_hid, lstm_hid, mlp_hid, gcn_hid, out_hid]
    alpha       : FTEncoder split ratio (used when pattern=1)
    pattern     : FTEncoder fusion pattern (0/1/2)
    num_layers  : LSTM layers
    num_keys    : vocabulary size (number of unique log event templates)
    drop        : dropout rate
    """
    def __init__(self, input_size, com_num, hidden_size, alpha, pattern,
                 num_layers, num_keys, drop=0.1):
        super().__init__()
        ft_hid, lstm_hid, mlp_hid, gcn_hid, out_hid = hidden_size

        self.lstm_hid = lstm_hid
        self.com_num  = com_num

        self.ftencoder  = FTEncoder(input_size, ft_hid, alpha, pattern)
        self.lstm_seq   = LSTMEncoder(ft_hid, lstm_hid, num_layers)   # sequence-level
        self.lstm_com   = LSTMEncoder(ft_hid, lstm_hid, num_layers)   # component-level
        self.irencoder  = IREncoder(lstm_hid, mlp_hid, gcn_hid, drop, com_num)

        self.att_fc = nn.Linear(lstm_hid, lstm_hid)
        self.fc1    = nn.Linear(2 * lstm_hid, out_hid)
        self.fc2    = nn.Linear(out_hid, num_keys)

        self.u_att = nn.Parameter(torch.zeros(1, lstm_hid))
        nn.init.xavier_uniform_(self.u_att.unsqueeze(0),
                                gain=nn.init.calculate_gain('relu'))

    def _resolve(self, per_x, per_index):
        """Group log embeddings by component index."""
        res = collections.OrderedDict()
        for i in range(per_x.shape[0]):
            key = per_index[i].item()
            res.setdefault(key, []).append(per_x[i])
        return res

    def _attention_pool(self, x):
        """Weighted sum over time dimension using learned query vector."""
        B, T, H = x.shape
        flat = x.reshape(B * T, H)
        scores = torch.mm(flat, self.u_att.T).reshape(B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)        # (B, T, 1)
        out = torch.sum(x * weights, dim=1)                     # (B, H)
        return F.relu(self.att_fc(out))

    def forward(self, x, index, _q_x, t_x):
        """
        x      : (B, W, emb_dim)   – sentence embeddings per window step
        index  : (B, W)            – component index per window step
        _q_x   : (B, num_keys)     – quantity pattern (unused in attention path)
        t_x    : (B, W)            – time deltas (seconds from window start)
        """
        # (a) Fuse embedding + time
        x = self.ftencoder((x, t_x))               # (B, W, ft_hid)

        # (b) Sequence-level LSTM
        seq_out = self.lstm_seq(x)                  # (B, lstm_hid)

        # (b+c+d) Component-level path
        batch_size = x.shape[0]
        com_outs = []
        for i in range(batch_size):
            res = self._resolve(x[i], index[i])     # {com_id: [emb, ...]}
            ac = []
            for com_id, embs in res.items():
                stack = torch.stack(embs).unsqueeze(0)  # (1, n_logs, ft_hid)
                ac.append(self.lstm_com(stack).squeeze(0))
            ac = torch.stack(ac)                         # (n_coms, lstm_hid)

            # (c) GCN relation only when >1 component present
            if ac.shape[0] > 1:
                ac = self.irencoder(ac, list(res.keys()))

            # (d) Attention pooling over components
            com_out = self._attention_pool(ac.unsqueeze(0))  # (1, lstm_hid)
            com_outs.append(com_out)

        com_out = torch.stack(com_outs).squeeze(1)  # (B, lstm_hid)

        out = F.relu(self.fc1(torch.cat((seq_out, com_out), -1)))
        return self.fc2(out)                         # (B, num_keys)
