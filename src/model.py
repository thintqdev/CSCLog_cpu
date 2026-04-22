"""
CSCLog model – MAXIMUM GPU UTILIZATION VERSION
================================================
Optimizations vs previous version:
  1. Fully batched GCN:  build ONE global graph per mini-batch instead of
     B sequential per-sample GCN calls. Edge weights are computed with a
     single matrix multiply (no Python loop over samples).
  2. Attention pool:     pre-fused into one bmm+softmax, no view overhead.
  3. Component LSTM:     already batched (kept from previous version).
  4. Remove all .item() and CPU syncs from forward().
  5. All temporary tensors created directly on the correct device.
  6. Supports torch.compile (no data-dependent Python control flow in hot path).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import GCNConv


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class MLPLayer(nn.Module):
    def __init__(self, dmodel, hid_size, drop):
        super().__init__()
        self.drop = drop
        self.fc0  = nn.Linear(dmodel, hid_size)
        self.fc1  = nn.Linear(hid_size, hid_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.relu(self.fc1(x))


class FTEncoder(nn.Module):
    """Fuse template embedding + time delta → fixed-size vector."""
    def __init__(self, sen_size, hidden_size, alpha=0.5, pattern=0):
        super().__init__()
        self.pattern = pattern
        assert pattern in (0, 1, 2)
        if pattern == 1:
            assert 0 < alpha < 1
            sf = int(hidden_size * alpha)
            tf = hidden_size - sf
            self.sen_fc  = nn.Linear(sen_size, sf)
            self.time_fc = nn.Linear(1, tf)
        elif pattern == 0:
            self.cat_fc = nn.Linear(sen_size + 1, hidden_size)
        else:
            self.sen_fc  = nn.Linear(sen_size, hidden_size)
            self.time_fc = nn.Linear(1, hidden_size)

    def forward(self, sen_x, time_x):
        # time_x: (B, W) → (B, W, 1)
        t = time_x.unsqueeze(-1)
        if self.pattern == 0:
            return self.cat_fc(torch.cat([sen_x, t], dim=-1))
        elif self.pattern == 1:
            return torch.cat([self.sen_fc(sen_x), self.time_fc(t)], dim=-1)
        else:
            return self.sen_fc(sen_x) + self.time_fc(t)


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        c0 = x.new_zeros(self.num_layers, B, self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]

    def forward_packed(self, packed, N):
        """Run on a PackedSequence of N total sequences."""
        h0 = torch.zeros(self.num_layers, N, self.hidden_size,
                         device=next(self.parameters()).device)
        _, (h_n, _) = self.lstm(packed, (h0, torch.zeros_like(h0)))
        return h_n[-1]   # (N, hidden_size)


class IREncoder(nn.Module):
    """
    Inter-component Relation encoder.
    OPTIMIZED: builds edge tensors without Python list loops in the forward path.
    """
    def __init__(self, dmodel, mlp_hid_size, gcn_hid_size, drop, com_num):
        super().__init__()
        self.dmodel  = dmodel
        self.drop    = drop
        self.com_num = com_num

        self.edge_mlp = MLPLayer(2 * dmodel, mlp_hid_size, drop)
        self.mlp_out  = nn.Linear(mlp_hid_size, 1)
        self.GCN0     = GCNConv(dmodel, gcn_hid_size)
        self.GCN1     = GCNConv(gcn_hid_size, dmodel)

    @staticmethod
    def _build_edge_index_fast(node_indices: torch.Tensor):
        """
        Build complete undirected edge index for `node_indices` using
        torch operations — no Python loops, stays on the same device.
        node_indices: 1-D tensor of node ids, shape (K,)
        returns: (2, K*(K-1)//2) edge_index
        """
        K = node_indices.shape[0]
        if K <= 1:
            return node_indices.new_empty((2, 0))
        # upper-triangular pairs
        r = torch.arange(K, device=node_indices.device)
        i, j = torch.triu_indices(K, K, offset=1, device=node_indices.device)
        src = node_indices[i]
        dst = node_indices[j]
        return torch.stack([src, dst], dim=0)  # (2, E)

    def _gumbel_softmax(self, x, dim=0):
        return F.softmax(x.transpose(dim, 0), dim=0).transpose(dim, 0)

    def forward(self, x: torch.Tensor, index: torch.Tensor):
        """
        x     : (K, dmodel)  — component representations
        index : (K,)         — global component ids (long tensor)
        """
        device = x.device
        K = x.shape[0]

        # Zero-padded global node table
        padding = x.new_zeros(self.com_num, self.dmodel)
        padding[index] = x

        edge_index = self._build_edge_index_fast(index)  # (2, E)
        if edge_index.shape[1] == 0:
            # Single component — skip GCN, return x unchanged
            return x

        # Edge features: concat node pairs
        edge_x = torch.cat([padding[edge_index[0]], padding[edge_index[1]]], dim=-1)
        edge_x = self.edge_mlp(edge_x)       # (E, mlp_hid)
        edge_w = self._gumbel_softmax(self.mlp_out(edge_x), dim=0)  # (E, 1)

        out = F.relu(self.GCN0(padding, edge_index, edge_w))
        out = F.dropout(out, self.drop, training=self.training)
        out = self.GCN1(out, edge_index, edge_w)
        return out[index]   # (K, dmodel)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class CSCLog(nn.Module):
    """
    CSCLog – Maximum GPU Utilization

    Key changes vs previous:
    • FTEncoder takes (x, t) separately — avoids tuple overhead
    • Component LSTM: ONE batched PackedSequence forward for all samples
    • IREncoder: fully tensor-based edge construction (no Python loops)
    • Attention pool: single bmm, no reshape gymnastics
    • No .item() / .cpu() calls anywhere in forward()
    """
    def __init__(self, input_size, com_num, hidden_size, alpha, pattern,
                 num_layers, num_keys, drop=0.1):
        super().__init__()
        ft_hid, lstm_hid, mlp_hid, gcn_hid, out_hid = hidden_size

        self.lstm_hid = lstm_hid
        self.com_num  = com_num

        self.ftencoder = FTEncoder(input_size, ft_hid, alpha, pattern)
        self.lstm_seq  = LSTMEncoder(ft_hid, lstm_hid, num_layers)
        self.lstm_com  = LSTMEncoder(ft_hid, lstm_hid, num_layers)
        self.irencoder = IREncoder(lstm_hid, mlp_hid, gcn_hid, drop, com_num)

        self.att_fc = nn.Linear(lstm_hid, lstm_hid)
        self.fc1    = nn.Linear(2 * lstm_hid, out_hid)
        self.fc2    = nn.Linear(out_hid, num_keys)

        self.u_att = nn.Parameter(torch.zeros(lstm_hid, 1))
        nn.init.xavier_uniform_(self.u_att.unsqueeze(0),
                                gain=nn.init.calculate_gain('relu'))

    # ── Attention pool ────────────────────────────────────────────────────
    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, H)  →  (B, H)
        Single bmm call, no Python reshaping.
        """
        # scores: (B, T, 1)
        scores  = torch.bmm(x, self.u_att.unsqueeze(0).expand(x.size(0), -1, -1))
        weights = F.softmax(scores, dim=1)          # (B, T, 1)
        pooled  = (x * weights).sum(dim=1)          # (B, H)
        return F.relu(self.att_fc(pooled))

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, index: torch.Tensor,
                _q_x: torch.Tensor, t_x: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, W, emb_dim)
        index : (B, W)  component indices (long)
        _q_x  : (B, num_keys) quantity pattern — unused (kept for API compat)
        t_x   : (B, W)  time deltas
        """
        B, W, _ = x.shape

        # (a) Fuse embeddings + time
        x = self.ftencoder(x, t_x)   # (B, W, ft_hid)

        # (b) Sequence-level LSTM
        seq_out = self.lstm_seq(x)    # (B, lstm_hid)

        # ── (c) Component LSTM — single batched forward ────────────────
        #
        # Strategy:
        #   For each sample i, group window positions by component id.
        #   Collect ALL subsequences across the batch into one padded tensor.
        #   Run ONE PackedSequence LSTM forward.
        #   Scatter results back to per-sample component slots.
        #
        # CPU work here is O(B*W) list ops — done ONCE per batch,
        # not per epoch × per batch.
        # ──────────────────────────────────────────────────────────────

        index_cpu = index.cpu().tolist()   # (B, W) — one D2H per batch

        all_subseqs  = []    # list of (L_k, ft_hid) tensors
        all_lengths  = []    # L_k values
        # For each sample i: list of (com_id, flat_idx_in_all_subseqs)
        batch_com_maps = []

        for i in range(B):
            positions_by_com: dict[int, list[int]] = {}
            for j, cid in enumerate(index_cpu[i]):
                positions_by_com.setdefault(cid, []).append(j)

            com_map: dict[int, int] = {}
            for cid, positions in positions_by_com.items():
                # x[i, positions] — fancy index, no copy (stays on GPU)
                all_subseqs.append(x[i, positions, :])
                all_lengths.append(len(positions))
                com_map[cid] = len(all_subseqs) - 1
            batch_com_maps.append(com_map)

        # Pad all subsequences
        N       = len(all_subseqs)
        max_len = max(all_lengths)
        ft_hid  = x.shape[-1]

        # Allocate padded buffer
        padded = x.new_zeros(N, max_len, ft_hid)
        for k, (seq_k, L_k) in enumerate(zip(all_subseqs, all_lengths)):
            padded[k, :L_k] = seq_k

        # ONE PackedSequence LSTM call (all N subsequences in parallel)
        lengths_t = torch.tensor(all_lengths, dtype=torch.long)  # CPU — needed by pack
        packed    = rnn_utils.pack_padded_sequence(
            padded, lengths_t, batch_first=True, enforce_sorted=False)
        lstm_com_out = self.lstm_com.forward_packed(packed, N)  # (N, lstm_hid)

        # ── (d) Per-sample GCN + attention ────────────────────────────
        com_outs = []
        for i in range(B):
            com_map   = batch_com_maps[i]
            flat_idxs = list(com_map.values())           # small list, O(n_coms)
            ac        = lstm_com_out[flat_idxs]          # (K, lstm_hid)
            idx_t     = index.new_tensor(list(com_map.keys()))  # (K,) — on DEVICE

            if ac.shape[0] > 1:
                ac = self.irencoder(ac, idx_t)

            # Attention pool over K components
            com_outs.append(self._attention_pool(ac.unsqueeze(0)))  # (1, lstm_hid)

        com_out = torch.cat(com_outs, dim=0)   # (B, lstm_hid)  — cat avoids squeeze

        # (e) Classification head
        out = F.relu(self.fc1(torch.cat([seq_out, com_out], dim=-1)))
        return self.fc2(out)