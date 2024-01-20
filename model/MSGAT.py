import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool, global_max_pool


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, use_sigmoid=False):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 8
        layers = [
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1)  # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=-1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention_block = AttentionBlock(time_step, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.dim = hidden_dim

    def forward(self, seq):
        '''
        inp : torch.tensor (batch,time_step,input_dim)
        '''
        seq_vector, _ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)
        attention_vec = attention_vec.view(-1, 1, self.dim)  # prepare for concat
        return attention_vec


class CategoricalGraphAtt(nn.Module):
    def __init__(self, input_dim, time_step_list, hidden_dim, inner_edge, outer_edge, len_array, use_gru, block_day,
                 device):
        super(CategoricalGraphAtt, self).__init__()

        # basic parameters
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.time_step_list = time_step_list
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.len_array = len_array
        self.use_gru = use_gru
        self.block_day = block_day
        self.device = device

        # hidden layers
        # self.pool_attention = AttentionBlock(25,hidden_dim)
        if self.use_gru:
            self.period_encoder = nn.GRU(hidden_dim, hidden_dim)

        self.cat_gat = GATConv(hidden_dim, hidden_dim)
        self.inner_gat = GATConv(hidden_dim, hidden_dim)

        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # output layer
        self.reg_layer = nn.Linear(hidden_dim, 1)
        self.cls_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x has shape (block_day, stocks_num, dim)
        fusion_reg = []
        fusion_cls = []
        # divide different period
        for time_step in self.time_step_list:
            # assume time_step = 6
            input_period_num = self.block_day // time_step  # 5
            start_day = self.block_day % time_step  # 2

            # delete extra day
            filter_x = x[start_day:].to(self.device)

            period_batch = []
            for period_idx in range(input_period_num):
                period_batch.append(filter_x[period_idx * time_step:(period_idx + 1) * time_step])

            encoder_list = nn.ModuleList(
                [SequenceEncoder(self.input_dim, time_step, self.hidden_dim) for _ in range(input_period_num)]
            ).to(self.device)

            period_embedding = encoder_list[0](period_batch[0].reshape(-1, time_step, self.input_dim))

            # calculate embeddings for the rest of weeks
            for period_idx in range(1, input_period_num):
                period_inp = period_batch[period_idx]
                period_inp = period_inp.reshape(-1, time_step, self.input_dim)
                period_stock_embedding = encoder_list[period_idx](period_inp)
                period_embedding = torch.cat((period_embedding, period_stock_embedding), dim=1)
            # print(f'after concat weekly_embedding size = {weekly_embedding.size()}')  # torch.Size([475, 3, 64])

            # merge weeks
            if self.use_gru:
                period_embedding, _ = self.period_encoder(period_embedding)

            period_attention = AttentionBlock(input_period_num, self.hidden_dim).to(self.device)
            period_att_vector, _ = period_attention(period_embedding)
            # print(f'weekly_att_vector size = {weekly_att_vector.size()}')  # torch.Size([475, 64])

            # inner graph interaction
            inner_graph_embedding = self.inner_gat(period_att_vector, self.inner_edge)
            # print(f'inner_graph_embedding size = {inner_graph_embedding.size()}')  # torch.Size([475, 64])

            # pooling
            start_index = 0
            category_vectors_list = []
            for i in range(len(self.len_array)):
                end_index = start_index + self.len_array[i]
                sector_graph_embedding = inner_graph_embedding[start_index:end_index, :].unsqueeze(0)
                pool_attention = AttentionBlock(self.len_array[i], self.hidden_dim).to(self.device)
                category_vectors, _ = pool_attention(sector_graph_embedding)  # ([1, 64])
                category_vectors_list.append(category_vectors)
                start_index = end_index

            category_vectors = torch.cat(category_vectors_list, dim=0)  # torch.max(weekly_att_vector,dim=1)
            # print(f'category_vectors size = {category_vectors.size()}')  # torch.Size([19, 64])

            # use category graph attention
            category_vectors = self.cat_gat(category_vectors, self.outer_edge)  # (5,dim)
            # print(f'after sector gat category_vectors size = {category_vectors.size()}')  # torch.Size([19, 64])

            intra_graph_embedding_list = []
            for i in range(category_vectors.size()[0]):
                gat_category_vectors = category_vectors[i:i + 1, :]
                for j in range(self.len_array[i]):
                    intra_graph_embedding_list.append(gat_category_vectors)
            intra_graph_embedding = torch.cat(intra_graph_embedding_list, dim=0)
            # print(f'intra_graph_embedding size = {intra_graph_embedding.size()}')   # torch.Size([475, 64])

            # fusion
            fusion_vec = torch.cat((period_att_vector, inner_graph_embedding, intra_graph_embedding), dim=-1)
            # print(f'cat fusion_vec size = {fusion_vec.size()}')  # torch.Size([475, 192])

            fusion_vec = self.fusion(fusion_vec)
            # print(f'linear fusion_vec size = {fusion_vec.size()}')  # torch.Size([475, 64])

            fusion_vec = torch.relu(fusion_vec)
            # print(f'relu fusion_vec size = {fusion_vec.size()}')  # torch.Size([475, 64])

            # output
            reg_output = self.reg_layer(fusion_vec)
            fusion_reg.append(reg_output)
            # print(f'reg_output size = {reg_output.size()}')  # torch.Size([475, 1])
            reg_output = torch.flatten(reg_output)
            # print(f'flatten reg_output size = {reg_output.size()}')  # torch.Size([475])

            cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
            fusion_cls.append(cls_output)
            cls_output = torch.flatten(cls_output)

        combined_reg = torch.cat(fusion_reg, dim=1)
        combined_cls = torch.cat(fusion_cls, dim=1)

        # 创建 MLP 实例
        reg_mlp = SimpleMLP(len(self.time_step_list)).to(self.device)
        cls_mlp = SimpleMLP(len(self.time_step_list), use_sigmoid=True).to(self.device)

        # 通过 MLP 处理合并的张量
        reg_predict = reg_mlp(combined_reg)
        reg_predict = torch.flatten(reg_predict)

        cls_predict = cls_mlp(combined_cls)
        cls_predict = torch.flatten(cls_predict)

        return reg_predict, cls_predict

    def predict_toprank(self, test_data, device, top_k=5):  # test_data : 384,480,10
        y_pred_all_reg, y_pred_all_cls = [], []
        # test_w1, test_w2, test_w3 = test_data
        for idx in range(test_data.shape[0] - self.block_day):
            # batch_x1, batch_x2, batch_x3 = test_w1[idx].to(self.device), \
            #     test_w2[idx].to(self.device), \
            #     test_w3[idx].to(self.device)
            # batch_weekly = [batch_x1, batch_x2, batch_x3]
            batch_x = test_data[idx:(idx + 32)]
            pred_reg, pred_cls = self.forward(batch_x)  # pred_reg, pred_cls : 480, 1
            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
            y_pred_all_reg.extend(pred_reg.tolist())
            y_pred_all_cls.extend(pred_cls.tolist())
        return y_pred_all_reg, y_pred_all_cls
