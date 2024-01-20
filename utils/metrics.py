import numpy as np
import pandas as pd


def MRR(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict = predict.sort_values("pred_y", ascending=False).reset_index(drop=True)
    predict["pred_y_rank_index"] = (predict.index) + 1
    predict = predict.sort_values("y", ascending=False)

    return sum(1 / predict["pred_y_rank_index"][:k])


def Precision(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict1 = predict.sort_values("pred_y", ascending=False)
    predict2 = predict.sort_values("y", ascending=False)
    correct = len(list(set(predict1["y"][:k].index) & set(predict2["y"][:k].index)))
    return correct / k


def IRR(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict1 = predict.sort_values("pred_y", ascending=False)
    predict2 = predict.sort_values("y", ascending=False)
    return sum(predict2["y"][:k]) - sum(predict1["y"][:k])


def Acc(test_y, pred_y):
    test_y = np.ravel(test_y)
    pred_y = np.ravel(pred_y)
    pred_y = (pred_y > 0) * 1
    acc_score = sum(test_y == pred_y) / len(pred_y)

    return acc_score