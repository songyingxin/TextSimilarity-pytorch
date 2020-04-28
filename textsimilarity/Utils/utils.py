# coding=utf-8

import time
from sklearn import metrics
import random
import numpy as np

import torch

def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def compute_metrics(preds, labels):
    """ 由于是二分类问题，因此采用二分类常见的几个评估指标：
    f1-score, acc, pre, recall"""

    acc = metrics.accuracy_score(labels, preds)
    # auc = metrics.roc_auc_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)

    result = {
        'acc': acc,
        # 'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return result

def softmax(x):
    """ 将结果转化为 softmax 结果 """
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x.tolist()