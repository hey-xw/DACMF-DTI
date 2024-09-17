import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, PDNConv, Set2Set, GRUAggregation, SetTransformerAggregation
import numpy as np
from torch.nn import init

#
# class NNConvNet(nn.Module):
#     def __init__(self, node_feature_dim, edge_feature_dim, edge_hidden_dim, latent_dim):
#         super(NNConvNet, self).__init__()
#         # 第一层
#         # edge_network1 = nn.Sequential(
#         #     nn.Linear(edge_feature_dim, edge_hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(edge_hidden_dim, node_feature_dim * node_feature_dim)
#         # )
#         self.gru1 = GRUAggregation(node_feature_dim, node_feature_dim)
#         self.gru2 = GRUAggregation(node_feature_dim, node_feature_dim)
#         # self.gru2 = SetTransformerAggregation(node_feature_dim)
#
#         # self.nnconv1 = NNConv(node_feature_dim, node_feature_dim, edge_network1, aggr="mean")
#         self.nnconv1 = PDNConv(node_feature_dim, node_feature_dim, edge_feature_dim, edge_hidden_dim, aggr='mean')
#         self.nnconv2 = PDNConv(node_feature_dim, node_feature_dim, edge_feature_dim, edge_hidden_dim, aggr='mean')
#         # # 第二层
#         # edge_network2 = nn.Sequential(
#         #     nn.Linear(edge_feature_dim, edge_hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(edge_hidden_dim, 32 * 16)
#         # )
#         # self.nnconv2 = NNConv(32, 16, edge_network2, aggr="mean")
#
#         self.dropout = nn.Dropout(0.2)
#         self.relu = nn.ReLU()
#         self.set2set = Set2Set(24, processing_steps=3)
#         self.fc2 = nn.Linear(node_feature_dim, latent_dim)
#         # self.fc3 = nn.Linear(8, 2)
#
#     def forward(self, atoms_vector, x_mask, edge_index, edge_attr, ):
#         # x, edge_index, edge_attr = atoms_vector, data.edge_index, data.edge_attr,
#         batch_zize = atoms_vector.shape[0]
#         # edge_index = edge_index_mask*edge_index
#         # edge_attr =edge_attr_mask*edge_attr
#         node = []
#         # batch = Batch.from_data_list([Data(x=x,edge_index=edge_index,edge_attr=edge_attr) for x,edge_index,edge_attr in zip(x,edge_index,edge_attr)])
#         for i in range(batch_zize):
#             # print(x[i].shape,edge_index[i].shape,edge_attr[i].shape)
#             n = self.nnconv1(atoms_vector[i], edge_index[i], edge_attr[i])
#             # n = self.gru1(n)
#             # n = self.relu(n)
#             n = self.nnconv2(n, edge_index[i], edge_attr[i])
#             # edge_index[i], edge_attr[i] = sort_edge_index(edge_index[i], edge_attr[i])
#             # n = self.gru2(n)
#             n = n * x_mask[i]
#             n = self.fc2(n)
#             n = self.dropout(n)
#             n = self.relu(n)
#
#             node.append(n)
#         x = torch.stack(node, dim=0)
#         # for i in range(batch_zize):
#         #     x = self.nnconv1(node[i], edge_index[i], edge_attr[i])
#         #     new_node.append(x)
#         # x = self.set2set(x, batch)
#         # 经过全连接层
#
#         return x
#
#
# class cross_layers(nn.Module):
#     def __init__(self, com_channel, pro_channel, channel, attention_dropout=0.05):
#         super(cross_layers, self).__init__()
#         # self.self_atten0 = self_attention(pro_channel)
#         # self.self_atten1 = self_attention(com_channel)
#         # self.cross_atten1 = cross_att(channel)
#         self.dropout = nn.Dropout(attention_dropout)
#         # self.avgpool1 = nn.AdaptiveAvgPool1d(1)
#         # self.avgpool2 = nn.AdaptiveAvgPool1d(1)
#         self.act = nn.ReLU()
#         self.pro_proj = nn.Sequential(
#             nn.Linear(channel, channel),
#             nn.ReLU()
#         )
#         self.com_proj = nn.Sequential(
#             nn.Linear(channel, channel),
#             nn.ReLU()
#         )
#         self.avgpool1 = nn.AdaptiveAvgPool1d(1)
#         self.avgpool2 = nn.AdaptiveAvgPool1d(1)
#         self.pro_proj2 = nn.Sequential(
#             nn.Linear(channel, channel),
#             nn.ReLU()
#         )
#         self.com_proj2 = nn.Sequential(
#             nn.Linear(channel, channel),
#             nn.ReLU()
#         )
#
#     def forward(self, pf, cf, pro, com):
#         pro = torch.cat((pf, pro), dim=1)
#         com = torch.cat((cf, com), dim=1)
#         pro = self.pro_proj(pro)
#         com = self.com_proj(com)
#         # pro = self.self_atten0(pro,pf)
#         # com = self.self_atten1(com,cf)
#         # pro = self.pro_proj2(pro)
#         # com = self.com_proj2(com)
#         # com1 = self.avgpool1(com.permute(0, 2, 1)).squeeze(2)
#         # pro1 = self.avgpool2(pro.permute(0, 2, 1)).squeeze(2)
#         d_max_len = com.shape[1]
#         d_max_feature_a = F.max_pool1d(com.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
#                                        padding=0, dilation=1, ceil_mode=False,
#                                        return_indices=False).squeeze(2)
#         d_avg_feature_a = F.avg_pool1d(com.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
#                                        padding=0, ceil_mode=False).squeeze(2)
#         drug_feature_a = torch.cat([d_max_feature_a, d_avg_feature_a], dim=1)
#
#         d_max_len = cf.shape[1]
#         d_max_feature = F.max_pool1d(cf.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
#                                      padding=0, dilation=1, ceil_mode=False,
#                                      return_indices=False).squeeze(2)
#         d_avg_feature = F.avg_pool1d(cf.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
#                                      padding=0, ceil_mode=False).squeeze(2)
#         drug_feature = torch.cat([d_max_feature, d_avg_feature], dim=1)
#
#         p_max_len = pro.shape[1]
#         p_max_feature_a = F.max_pool1d(pro.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
#                                        padding=0, dilation=1, ceil_mode=False,
#                                        return_indices=False).squeeze(2)
#         p_avg_feature_a = F.avg_pool1d(pro.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
#                                        padding=0, ceil_mode=False).squeeze(2)
#         protein_feature_a = torch.cat([p_max_feature_a, p_avg_feature_a], dim=1)
#
#         p_max_len = pf.shape[1]
#         p_max_feature = F.max_pool1d(pf.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
#                                      padding=0, dilation=1, ceil_mode=False,
#                                      return_indices=False).squeeze(2)
#         p_avg_feature = F.avg_pool1d(pf.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
#                                      padding=0, ceil_mode=False).squeeze(2)
#         protein_feature = torch.cat([p_max_feature, p_avg_feature], dim=1)
#         #
#         iner_f = torch.mul(drug_feature_a, protein_feature_a)
#         pair = torch.cat([drug_feature, iner_f, protein_feature], dim=1)
#
#         # pc_fac = self.cross_atten1(pro1, com1)
#
#         return iner_f


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * self.softmax(x) + x


# class self_attention(nn.Module):
#     def __init__(self, dim, num_heads):
#         super(self_attention, self).__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.proj_q1 = nn.Linear(dim, dim * num_heads, bias=False)
#         self.proj_k2 = nn.Linear(dim, dim * num_heads, bias=False)
#         self.proj_v2 = nn.Linear(dim, dim * num_heads, bias=False)
#         self.proj_o = nn.Linear(dim * num_heads, dim)
#
#     def forward(self, x1, x2, mask=None):
#         batch_size, seq_len1, in_dim1 = x1.size()
#         seq_len2 = x2.size(1)
#
#         q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.dim).permute(0, 2, 1, 3)
#         k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.dim).permute(0, 2, 3, 1)
#         v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.dim).permute(0, 2, 1, 3)
#
#         attn = torch.matmul(q1, k2) / self.dim ** 0.5
#
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
#
#         attn = F.softmax(attn, dim=-1)
#         output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
#         output = self.proj_o(output)
#
#         return output


# class cross_att(nn.Module):
#     def __init__(self, channel):
#         super(cross_att, self).__init__()
#         # self.att0_2 = atten(channel)
#         self.att1 = atten(channel)
#         self.linear_channel1 = nn.Linear(channel*channel, 2)
#         self.act = Swish(True)
#         #self.linear_channel2 = nn.Linear(channel, channel)
#         #self.linear_length1 = nn.Linear(length, length)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(0.4)
#
#     def forward(self, mtrA, mtrB):
#         b = mtrA.shape[0]
#         cf_pf = self.act(torch.matmul(mtrA.view(b, -1, 1), mtrB.view(b, 1, -1)).view(b, -1))
#         cf_pf = self.linear_channel1(cf_pf)
#         cf_pf = self.dropout(cf_pf)
#         return cf_pf

# class mutil_head_attention(nn.Module):
#     def __init__(self,head = 8,conv=32):
#         super(mutil_head_attention,self).__init__()
#         self.conv = conv
#         self.head = head
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
#         self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
#         self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).cuda()
#
#     def forward(self, drug, protein):
#         bsz, d_ef,d_il = drug.shape
#         bsz, p_ef, p_il = protein.shape
#         drug_att = self.relu(self.d_a(drug.permute(0,2,1))).view(bsz,self.head,d_il,d_ef)
#         protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz,self.head,p_il,p_ef)
#         interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
#         Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
#         Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
#         drug = drug * Compound_atte
#         protein = protein * Protein_atte
#         # drug = drug.reshape(bsz,-1)
#         # protein = protein.reshape(bsz,-1)
#         return drug,protein
#

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_k, d_v, d_model, h=4, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.fc_q = nn.Linear(d_k,  h * d_k)
        self.fc_k = nn.Linear(d_k,  h * d_k)
        self.fc_p = nn.Linear(d_v,  h * d_v)
        self.fc_g = nn.Linear(d_v,  h * d_v)
        self.fc_o1 = nn.Linear(h * d_k, d_k)
        self.fc_o2 = nn.Linear(h * d_v, d_v)
        #
        self.fc_3 = nn.Sequential(
            nn.Linear(d_v, d_v * 2),
            nn.ReLU(),
            nn.Linear(d_v * 2, d_v),
            nn.ReLU()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear(d_v, d_v * 2),
            nn.ReLU(),
            nn.Linear(d_v * 2, d_v),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        u = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        v = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        p = self.fc_p(keys).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        g = self.fc_g(queries).view(b_s, nq, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nq, d_v)

        att = torch.matmul(u, v) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        # if pro_mask is not None:
        #     pro_att = att.masked_fill(pro_mask, -np.inf)
        att_p = torch.softmax(att, -1)
        att_p = self.dropout(att_p)

        P = torch.matmul(att_p, p).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)

        # if drug_mask is not None:
        #     drug_att = att.masked_fill(drug_mask, -np.inf)
        att_d = torch.softmax(att.permute(0, 1, 3, 2), -1)
        att_d = self.dropout(att_d)
        D = torch.matmul(att_d, g).permute(0, 2, 1, 3).contiguous().view(b_s, nk, self.h * self.d_v)  # (b_s, nq, h*d_v)

        P = self.fc_o1(P)  # (b_s, nq, d_model)
        D = self.fc_o2(D)  #(b_s, nq, d_model)
        # P = torch.cat([P, keys], dim=1)
        # D = torch.cat([D, queries], dim=1)
        #
        # P = self.fc_3(P)
        # D = self.fc_4(D)

        return P,D

class AttentionDTA(nn.Module):
    def __init__(self, protein_MAX_LENGH=1200, protein_kernel=[4, 8, 12],
                 drug_MAX_LENGH=100, drug_kernel=[4, 6, 8],
                 conv=64, char_dim=128, head_num=8, dropout_rate=0.1):
        super(AttentionDTA, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel
        # node_feature_dim, edge_feature_dim, edge_hidden_dim = 45, 6, 32

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        # self.Drug_max_pool = nn.MaxPool1d(
        #     self.drug_MAX_LENGH - self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3)
        # self.Drug_max_pool = nn.AdaptiveMaxPool1d(1)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        # self.Protein_max_pool = nn.MaxPool1d(
        #     self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        # self.mpnn = NNConvNet(node_feature_dim, edge_feature_dim, edge_hidden_dim,self.conv*3)

        # self.cross_layers = cross_layers(self.conv * 3, self.conv * 3, self.conv * 3)
        # self.attention = mutil_head_attention(head = self.head_num, conv=self.conv)
        self.cross_modal = ScaledDotProductAttention(self.conv * 3, self.conv * 3, self.conv * 3,   head_num)
        # self.cross_attention = self_attention(self.conv * 3)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        # self.fc1 = nn.Linear(576, 1024)
        # self.dropout1 = nn.Dropout(self.dropout_rate)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.dropout2 = nn.Dropout(self.dropout_rate)
        # self.fc3 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, 2)
        # torch.nn.init.constant_(self.out.bias, 5)
        # self.fc_o3 = nn.Sequential(
        #     nn.Linear(self.conv * 6, self.conv * 6),
        #     nn.ReLU()
        # )
        # self.fc_o4 = nn.Sequential(
        #     nn.Linear(self.conv * 6, self.conv * 6),
        #     nn.ReLU()
        # )
        # self.pro_proj = nn.Sequential(
        #     nn.Linear(self.conv * 3, self.conv * 3),
        #     nn.ReLU()
        # )
        # self.com_proj = nn.Sequential(
        #     nn.Linear(self.conv * 3, self.conv * 3),
        #     nn.ReLU()
        # )

        self.fc = nn.Sequential(
            nn.Linear(self.conv * 6, self.conv * 6),
            nn.ReLU()
        )
        self.self_attention = nn.MultiheadAttention(self.conv * 6,4)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv*12, 1024),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def fu_attention(self, atoms_vector, amino_vector, compound_mask, protein_mask):
        amino_vector = amino_vector.permute(0, 2, 1)
        atoms_vector = atoms_vector.permute(0, 2, 1)
        P,D = self.cross_modal(atoms_vector, amino_vector)
        P = torch.cat([amino_vector,P],dim=1)
        D = torch.cat([D,atoms_vector],dim=1)
        # P = self.pro_proj(P)
        # D = self.com_proj(D)
        iner = torch.cat([P,D],dim = 2)
        iner = self.fc(iner)
        # query = P
        # key = D
        # x, _ = self.attention(query, key, key)
        # x = x + P
        # x = x.transpose(0, 1)

        iner_f,_ = self.self_attention(iner,iner,iner)
        iner = iner_f + iner
        # P, D = self.cross_modal(D, P)
        # P = self.cross_attention(P,amino_vector)
        # D = self.cross_attention(D, atoms_vector)
        #
        # P = P + amino_vector
        # D = D + atoms_vector

        # pair = self.cross_layers(pf, cf, amino_vector, atoms_vector)
        # cf_pf = F.leaky_relu(torch.matmul(dta1.view(b, -1, 1), dta2.view(b, 1, -1)).view(b, -1), 0.1)
        # 把这两个向量对应打元素相乘，结果加起来，再整形为（b,-1）  过一个激活函数
        return iner

    def forward(self, drug, protein, atom, node, edge_index, edge_attr, compound_mask, protein_mask):
        drugembed = self.drug_embed(drug)
        # atomembed = self.drug_embed(atom)

        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # drugConv ,proteinConv = self.attention(drugConv,proteinConv)
        # atom = self.mpnn(node,atom,edge_index,edge_attr)
        # drug ,protein = self.attention(drugConv,proteinConv)
        pair = self.fu_attention(drugConv, proteinConv, compound_mask, protein_mask)

        d_max_len = pair.shape[1]
        d_max_feature_a = F.max_pool1d(pair.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
                                       padding=0, dilation=1, ceil_mode=False,
                                       return_indices=False).squeeze(2)
        d_avg_feature_a = F.avg_pool1d(pair.permute(0, 2, 1), kernel_size=d_max_len, stride=1,
                                       padding=0, ceil_mode=False).squeeze(2)
        drug_feature_a = torch.cat([d_max_feature_a, d_avg_feature_a], dim=1)

        # p_max_len = P.shape[1]
        # p_max_feature_a = F.max_pool1d(P.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
        #                                padding=0, dilation=1, ceil_mode=False,
        #                                return_indices=False).squeeze(2)
        # p_avg_feature_a = F.avg_pool1d(P.permute(0, 2, 1), kernel_size=p_max_len, stride=1,
        #                                padding=0, ceil_mode=False).squeeze(2)
        # protein_feature_a = torch.cat([p_max_feature_a, p_avg_feature_a], dim=1)
        # drug_feature_a = self.fc_o3(drug_feature_a)
        # protein_feature_a = self.fc_o4(protein_feature_a)
        # iner_f = torch.cat([drug_feature_a,protein_feature_a],dim=1)
        # iner_f = torch.mul(drug_feature_a, protein_feature_a)
        # drugConv ,proteinConv = self.attention(drugConv,proteinConv)
        # drugConv ,proteinConv = self.fu_attention(drugConv,proteinConv)
        # drugConv = self.Drug_max_pool(drugConv.permute(0,2,1)).squeeze(2)
        # proteinConv = self.Protein_max_pool(proteinConv.permute(0,2,1)).squeeze(2)
        # drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        # proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        # pair = torch.cat([drugConv,proteinConv], dim=1)
        predict = self.classifier(drug_feature_a)
        return predict
