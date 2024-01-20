import copy
import json
import os
import pickle
import random

import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from torch import optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from model.MSGAT import CategoricalGraphAtt
from utils.metrics import MRR, Precision, IRR, Acc
from layers.Fourier import FFT_for_Period


# class StockDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
#
# def create_dataloader(x, y, batch_size):
#     dataset = StockDataset(x, y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader

def train():
    block_day = 32
    # load data
    data_path = './datasets/sp500_data2.pkl'
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    inner_edge = np.array(np.load("./datasets/inner_edge.npy"))
    outer_edge = np.array(np.load("./datasets/outer_edge.npy"))

    num_weeks = data["train"]["x"].shape[-2]  # 1536
    input_dim = data["train"]["x"].shape[-1]  # 10

    lists = []
    weight_matrices = []
    for i in range(num_weeks // block_day):
        x = torch.tensor(data['train']['x'][:, i * block_day:(i + 1) * block_day, :])
        period, weight = FFT_for_Period(x, k=6)
        lists.append(period)
        weight_matrices.append(weight)

    # 将所有列表合并成一个大列表
    all_elements = []
    for lst in lists:
        all_elements.extend(lst)

    # 计算每个元素的出现次数
    element_counts = Counter(all_elements)

    # 计算每个元素的总 weight
    total_weights = {element: 0 for element in element_counts}
    for lst, matrix in zip(lists, weight_matrices):
        for i, element in enumerate(lst):
            total_weights[element] += torch.sum(matrix[:, i]).item()  # 使用torch.sum()并将结果转换为Python标量

    # 打印结果
    for element, count in element_counts.items():
        print(f"元素 {element} 出现了 {count} 次，总 weight 是 {total_weights[element]}")

    # 获取最终的period_list
    time_step_list = list(element_counts.keys())[1:-1]
    period_list = random.sample(time_step_list, 4)

    train_size = int(num_weeks * 0.2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # convert data into torch dtype
    train_x = torch.Tensor(data['train']['x']).float().permute(1, 0, 2).to(device)  # 1536,480,10
    # train_w1 = torch.Tensor(data["train"]["x1"][:, :, :, 1:]).float().to(device)  # 1578,480,7,10
    # train_w2 = torch.Tensor(data["train"]["x2"][:, :, :, 1:]).float().to(device)  # 1578,480,7,10
    # train_w3 = torch.Tensor(data["train"]["x3"][:, :, :, 1:]).float().to(device)  # 1578,480,7,10

    inner_edge = torch.tensor(inner_edge.T, dtype=torch.int64).to(device)
    outer_edge = torch.tensor(outer_edge.T, dtype=torch.int64).to(device)

    # test data
    test_x = torch.Tensor(data['test']['x']).float().permute(1, 0, 2).to(device)  # 384,480,10
    # test_w1 = torch.Tensor(data["test"]["x1"][:, :, :, 1:]).float().to(device)  # 388,480,7,10
    # test_w2 = torch.Tensor(data["test"]["x2"][:, :, :, 1:]).float().to(device)  # 388,480,7,10
    # test_w3 = torch.Tensor(data["test"]["x3"][:, :, :, 1:]).float().to(device)  # 388,480,7,10
    # test_data = [test_w1, test_w2, test_w3]  # [-agg_week_num:]

    # label data
    train_reg = torch.Tensor(data["train"]["y_return ratio"]).float()
    train_cls = torch.Tensor(data["train"]["y_up_or_down"]).float()
    test_y = data["test"]["y_return ratio"]
    test_cls = data["test"]["y_up_or_down"]
    test_shape = test_y.shape[0]
    loop_number = 100
    ks_list = [5, 10, 20]

    sp = pd.read_csv("./datasets/SP500_Companies.csv", encoding='ISO-8859-1')
    sector_list = sp["Sector"].unique()
    len_list = []
    for sector in sector_list:
        len_list.append(len(sp[sp["Sector"] == sector]))
    len_array = np.array(len_list)

    l2 = 0.1
    lr = 0.1
    beta = 0.1
    gamma = 0.1
    alpha = 0.1
    epochs = 50
    hidden_dim = 64
    use_gru = False

    model = CategoricalGraphAtt(input_dim, period_list, hidden_dim, inner_edge, outer_edge, len_array, use_gru, block_day, device).to(device)

    # initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:%s" % pytorch_total_params)

    # optimizer & loss
    optimizer = optim.Adam(model.parameters(), weight_decay=l2, lr=lr)
    reg_loss_func = nn.L1Loss(reduction="none")
    cls_loss_func = nn.BCELoss(reduction="none")

    # save best model
    best_metric_IRR = None
    best_metric_MRR = None
    best_results_IRR = None
    best_results_MRR = None
    global_best_IRR = 999
    global_best_MRR = 0

    r_loss = torch.tensor([]).float().to(device)
    c_loss = torch.tensor([]).float().to(device)
    ra_loss = torch.tensor([]).float().to(device)


    for epoch in range(epochs):
        for idx in range(num_weeks//block_day):
            model.train()  # prep to train model
            batch_x = train_x[idx * block_day:(idx+1)*block_day].to(device)
            batch_reg_y = train_reg[idx * block_day:(idx + 1) * block_day].view(-1, 1).to(device)
            batch_cls_y = train_cls[idx * block_day:(idx + 1) * block_day].view(-1, 1).to(device)
            # batch_x1, batch_x2, batch_x3 = (
            #     train_w1[week].to(device),
            #     train_w2[week].to(device),
            #     train_w3[week].to(device)
            # )
            # batch_weekly = [batch_x1, batch_x2, batch_x3]

            reg_out, cls_out = model(batch_x)
            reg_out, cls_out = reg_out.view(-1, 1), cls_out.view(-1, 1)

            # calculate loss
            reg_loss = reg_loss_func(reg_out, batch_reg_y)  # (target_size, 1)
            cls_loss = cls_loss_func(cls_out, batch_cls_y)
            rank_loss = torch.relu(
                -(reg_out.view(-1, 1) * reg_out.view(1, -1)) * (batch_reg_y.view(-1, 1) * batch_reg_y.view(1, -1)))
            c_loss = torch.cat((c_loss, cls_loss.view(-1, 1)))
            r_loss = torch.cat((r_loss, reg_loss.view(-1, 1)))
            ra_loss = torch.cat((ra_loss, rank_loss.view(-1, 1)))

            if (week + 1) % 1 == 0:
                cls_loss = beta * torch.mean(c_loss)
                reg_loss = alpha * torch.mean(r_loss)
                rank_loss = gamma * torch.sum(ra_loss)
                loss = reg_loss + rank_loss + cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                r_loss = torch.tensor([]).float().to(device)
                c_loss = torch.tensor([]).float().to(device)
                ra_loss = torch.tensor([]).float().to(device)
                if (week + 1) % 20 == 0:
                    print(f"epoch {epoch}, week {week} : REG Loss:%.4f CLS Loss:%.4f RANK Loss:%.4f  Loss:%.4f" % (
                    reg_loss.item(), cls_loss.item(), rank_loss.item(), loss.item()))

        # evaluate
        model.eval()
        print("Evaluate at epoch %s" % (epoch + 1))
        y_pred, y_pred_cls = model.predict_toprank([test_w1, test_w2, test_w3], device, top_k=5)

        # calculate metric
        y_pred = np.array(y_pred).ravel()
        test_y = np.array(test_y).ravel()
        mae = round(mean_absolute_error(test_y, y_pred), 4)
        acc_score = Acc(test_cls, y_pred)

        results = []
        for k in ks_list:
            IRRs, MRRs, Prs = [], [], []
            for i in range(test_shape):
                M = MRR(np.array(test_y[loop_number * i: loop_number * (i + 1)]),
                        np.array(y_pred[loop_number * i: loop_number * (i + 1)]), k=k)
                MRRs.append(M)
                P = Precision(
                    np.array(test_y[loop_number * i: loop_number * (i + 1)]),
                    np.array(y_pred[loop_number * i: loop_number * (i + 1)]), k=k
                )
                Prs.append(P)
            over_all = [mae, round(acc_score, 4), round(np.mean(MRRs), 4), round(np.mean(Prs), 4)]
            results.append(over_all)
        print(results)

        # print('MAE:',round(mae,4),' IRR:',round(np.mean(IRRs),4),' MRR:',round(np.mean(MRRs),4)," Precision:",round(np.mean(Prs),4))
        performance = [round(mae, 4), round(acc_score, 4), round(np.mean(MRRs), 4), round(np.mean(Prs), 4)]

        # print(performance)

        # save best
        if np.mean(MRRs) > global_best_MRR:
            global_best_MRR = np.mean(MRRs)
            best_metric_MRR = performance
            best_results_MRR = results

    return best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR


if __name__ == "__main__":
    ks_list = [5, 10, 20]
    best_metric_IRR, best_metric_MRR, best_results_IRR, best_results_MRR = train()
    print("-------Final result-------")
    print("[BEST MRR] MAE:%.4f ACC:%.4f MRR:%.4f Precision:%.4f" % tuple(best_metric_MRR))
    for idx, k in enumerate(ks_list):
        print("[BEST RESULT MRR with k=%s] MAE:%.4f ACC:%.4f MRR:%.4f Precision:%.4f" % tuple(
            tuple([k]) + tuple(best_results_MRR[idx])))
