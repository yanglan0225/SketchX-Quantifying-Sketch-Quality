import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=5, hidden_size=config.d_feed,
                            num_layers=config.n_layers, bidirectional=True,
                            batch_first=True, dropout=config.dropout)

        self.classifer = nn.Linear(config.d_feed, config.num_classes)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(config.d_feed)
        self.fc = nn.Linear(config.d_feed * 2, config.d_feed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, lens):

        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        input_x = x.index_select(0, Variable(idx_sort))
        length_list = list(lens[idx_sort])
        pack = nn.utils.rnn.pack_padded_sequence(input_x, length_list, batch_first=True)
        out, state = self.lstm(pack)
        un_padded = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        un_padded = un_padded[0].index_select(0, Variable(idx_unsort))
        out = self.dropout(un_padded)

        lens_idx = lens - 1
        lens_list = lens_idx.cpu().int().tolist()

        feats = out[torch.arange(x.size(0)), lens_list, :]

        return self.activation(self.fc(self.dropout(feats)))

class GACL(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        """
        :param input_dim:
        :param output_dim: number of class
        :param scale:
        W -> W/||W||
        """
        super(GACL, self).__init__()

        self.u_a = config.u_a
        self.u_m = config.u_m
        self.l_a = config.l_a
        self.l_m = config.l_m
        k = (self.u_m - self.l_m) / (self.u_a - self.l_a)
        self.min_lamada = config.scale * k * self.u_a ** 2 * self.l_a ** 2 / (self.u_a ** 2 - self.l_a ** 2)
        self.weight = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        #nn.init.xavier_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.weight)
        self.scale = config.scale

        self.epoch = 10
        self.tau = config.tau
        self.instant = config.GACL

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def get_margin(self, x):
        margin = (self.u_m - self.l_m) / \
                 (self.u_a - self.l_a) * (x - self.l_a) + self.l_m
        return margin

    def forward(self, input, target):

        batch_size = input.size(0)

        x_norm = torch.norm(input, 2, 1)#.clamp(self.l_a, self.u_a)

        loss_g = self.calc_loss_G(x_norm) # calculate g(a)

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1, 1)
        target_cos = cos_theta[range(batch_size), target]

        ada_margin = self.get_margin(x_norm)  # calculate m(a)
        adjusted_margin = ada_margin
        adjusted_margin = adjusted_margin.clamp(self.l_m, self.u_m)
        if self.instant == 'm1':
            GACL_cos = self.m1(target_cos, adjusted_margin)
        if self.instant == 'm2':
            GACL_cos = self.m2(target_cos, adjusted_margin)
        if self.instant == 'm3':
            GACL_cos = self.m3(target_cos, adjusted_margin)
        if self.instant == 'm4':
            GACL_cos = self.m4(target_cos, adjusted_margin)

        if self.training:
            preds_ = cos_theta
            preds_[range(batch_size), target] = torch.squeeze(GACL_cos)  # replace the y_i from (cos theta) to (cos theta + m)

            return self.scale * preds_, loss_g, x_norm
        else:
            preds_ = cos_theta/self.scale
            return preds_, loss_g, x_norm, preds_[range(batch_size), target]

    def m1(self, cos_theta, margin):
        return (1-margin)*cos_theta

    def m2(self, cos_theta, margin):
        theta = torch.arccos(cos_theta)
        return torch.cos(margin*theta)

    def m3(self, cos_theta, margin):
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_margin = torch.cos(margin)
        sin_margin = torch.sin(margin)

        return cos_theta*cos_margin - sin_theta*sin_margin

    def m4(self, cos_theta, margin):
        return cos_theta - margin