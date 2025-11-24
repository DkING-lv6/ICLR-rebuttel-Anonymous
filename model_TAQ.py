import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(Model, self).__init__()

        self.gcn1 = GCN(n_in, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        self.act = nn.ReLU()
        self.fc4 = nn.Linear(n_h, n_h, bias=False)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, seq1, adj, sparse=False):
        h_1 = self.gcn1(seq1, adj, sparse)
        h_1 = self.dropout(h_1)
        emb = self.gcn2(h_1, adj, sparse)
        f_1 = self.fc1(emb)
        f_1 = self.act(f_1)
        f_2 = self.fc2(f_1)
        f_2 = self.act(f_2)
        f_2 = self.dropout(f_2)
        logits = self.fc3(f_2)

        return emb, logits




