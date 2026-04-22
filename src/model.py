import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import GCNConv


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
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, N, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, N, self.hidden_size, device=device)
        _, (h_n, _) = self.lstm(packed, (h0, c0))
        return h_n[-1]


class IREncoder(nn.Module):
    def __init__(self, dmodel, mlp_hid_size, gcn_hid_size, drop, com_num):
        super().__init__()
        self.dmodel  = dmodel
        self.drop    = drop
        self.com_num = com_num

        self.edge_mlp = MLPLayer(2 * dmodel, mlp_hid_size, drop)
        self.mlp_out  = nn.Linear(mlp_hid_size, 1)
        # add_self_loops=True ensures every node has degree >= 1 -> no 1/sqrt(0) NaN
        self.GCN0     = GCNConv(dmodel, gcn_hid_size, add_self_loops=True)
        self.GCN1     = GCNConv(gcn_hid_size, dmodel, add_self_loops=True)

    def forward(self, x: torch.Tensor, index: torch.Tensor):
        K = x.shape[0]
        if K <= 1:
            return x

        # LOCAL indices 0..K-1 — GCNConv only sees K nodes, all connected
        # This is the root fix: previous code passed com_num-sized padding
        # where most nodes had degree=0 -> 1/sqrt(0) = NaN in GCN normalization
        local_i, local_j = torch.triu_indices(K, K, offset=1, device=x.device)
        edge_index = torch.stack([
            torch.cat([local_i, local_j]),
            torch.cat([local_j, local_i])
        ], dim=0)  # (2, K*(K-1))

        src_feat = x[edge_index[0]]
        dst_feat = x[edge_index[1]]
        edge_x   = torch.cat([src_feat, dst_feat], dim=-1)
        edge_x   = self.edge_mlp(edge_x)
        edge_w   = torch.sigmoid(self.mlp_out(edge_x)).clamp(min=1e-6, max=1.0)

        out = F.relu(self.GCN0(x, edge_index, edge_w))
        out = F.dropout(out, self.drop, training=self.training)
        out = self.GCN1(out, edge_index, edge_w)
        return out  # (K, dmodel)


class CSCLog(nn.Module):
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

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        scores  = torch.bmm(x, self.u_att.unsqueeze(0).expand(x.size(0), -1, -1))
        scores  = scores.clamp(-30.0, 30.0)
        weights = F.softmax(scores, dim=1)
        pooled  = (x * weights).sum(dim=1)
        return F.relu(self.att_fc(pooled))

    def forward(self, x: torch.Tensor, index: torch.Tensor,
                _q_x: torch.Tensor, t_x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape

        index = index.clamp(0, self.com_num - 1)

        x = self.ftencoder(x, t_x)
        seq_out = self.lstm_seq(x)

        index_cpu    = index.cpu().tolist()
        all_subseqs  = []
        all_lengths  = []
        batch_com_maps = []

        for i in range(B):
            positions_by_com: dict = {}
            for j, cid in enumerate(index_cpu[i]):
                positions_by_com.setdefault(cid, []).append(j)

            com_map: dict = {}
            for cid, positions in positions_by_com.items():
                all_subseqs.append(x[i, positions, :])
                all_lengths.append(len(positions))
                com_map[cid] = len(all_subseqs) - 1
            batch_com_maps.append(com_map)

        N       = len(all_subseqs)
        max_len = max(all_lengths)
        ft_hid  = x.shape[-1]

        padded = x.new_zeros(N, max_len, ft_hid)
        for k, (seq_k, L_k) in enumerate(zip(all_subseqs, all_lengths)):
            padded[k, :L_k] = seq_k

        lengths_t    = torch.tensor(all_lengths, dtype=torch.long)
        packed       = rnn_utils.pack_padded_sequence(
            padded, lengths_t, batch_first=True, enforce_sorted=False)
        lstm_com_out = self.lstm_com.forward_packed(packed, N)

        com_outs = []
        for i in range(B):
            com_map   = batch_com_maps[i]
            flat_idxs = list(com_map.values())
            ac        = lstm_com_out[flat_idxs]
            idx_t     = index.new_tensor(list(com_map.keys()))

            if ac.shape[0] > 1:
                ac = self.irencoder(ac, idx_t)

            com_outs.append(self._attention_pool(ac.unsqueeze(0)))

        com_out = torch.cat(com_outs, dim=0)

        out = F.relu(self.fc1(torch.cat([seq_out, com_out], dim=-1)))
        return self.fc2(out)