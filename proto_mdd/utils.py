import os
import shutil
import time
import pprint
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

def log(log_file_path, string):
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
                shutil.rmtree(path)
                os.mkdir(path)
    else:
        os.mkdir(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cosine_metric(a,b):
    # a: query feature
    # b: prototype
    n = a.shape[0]
    m = b.shape[0]
    tmp_a = a
    tmp_b = b
    tmp_a = tmp_a.unsqueeze(1).expand(n, m, -1)
    tmp_b = tmp_b.unsqueeze(0).expand(n, m, -1)
    logits = -((tmp_a - tmp_b)**2).sum(dim=2) # calculate the euclidean distance just as the threshold
    for i in range(n):
        logits[i, ] = torch.cosine_similarity(torch.unsqueeze(a[i,], dim =0), b)

    return logits



class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def compute_domain_gap(fea_src_shot, fea_src_query, fea_tgt_shot, fea_tgt_query, args):    
    # compute the distance between two domains
    proto_src = fea_src_shot.reshape(args.shot, -1, fea_src_shot.shape[-1]).mean(dim = 0) # [args.way, fea_dim]
    proto_tgt = fea_tgt_shot.reshape(args.shot, -1, fea_tgt_shot.shape[-1]).mean(dim = 0) # [args.way, fea_dim]
    proto = torch.cat([proto_src, proto_tgt], axis = 0) # [args.way * 2, fea_dim]
    query_src_logit_euclidean = euclidean_metric(fea_src_query, proto)
    query_tgt_logit_euclidean = euclidean_metric(fea_tgt_query, proto)

    query_src_logit_cosine = cosine_metric(fea_src_query, proto)
    query_tgt_logit_cosine = cosine_metric(fea_tgt_query, proto)

    pred_labels_query_src = query_src_logit_euclidean.max(1)[1]
    pred_labels_query_tgt = query_tgt_logit_euclidean.max(1)[1]
    
    criterion = nn.CrossEntropyLoss()
    mdd_loss_src = criterion(query_src_logit_cosine, pred_labels_query_src)
    logloss_tgt = torch.log(1 - F.softmax(query_tgt_logit_cosine, dim = 1))
    mdd_loss_tgt = F.nll_loss(logloss_tgt, pred_labels_query_tgt)

    mdd_loss = args.srcweight * mdd_loss_src + mdd_loss_tgt
    
    # print('mdd_loss on source and target domain: {:.4f}-{:.4f}\n'.format(mdd_loss_src.item(), mdd_loss_tgt.item()))
    return mdd_loss





