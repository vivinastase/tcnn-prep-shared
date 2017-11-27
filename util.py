import itertools

from scipy import stats
import torch
from torch.autograd import Variable

import constants
import numpy as np


cache = dict()
def chunk(arr,chunk_size):
    if len(arr)==0:
        yield []

    for i in range(0,len(arr),chunk_size):
        yield arr[i:i+chunk_size]

def sample(data,num_samples,replace=False):

    if len(data) <= num_samples:
        return data
    idx = np.random.choice(len(data),num_samples,replace=replace)
    return [data[i] for i in idx]


def ranks(scores, ascending = False):
    sign = 1 if ascending else -1
    scores = scores * sign
    ranks = stats.rankdata(scores)
    return ranks

def quantile(rank,total):
    '''
    Returns quantile or how far from the top rank / total
    :param rank:
    :param total:
    :return quantile:
    '''
    if total==1:
        return np.nan
    return (total-rank)/float(total-1)

def rank_from_quantile(quantile, total):
    if np.isnan(quantile):
        return total-1
    return total - quantile * (total - 1)



# Assumes number of positives and negatives
def average_quantile(positives,negatives):
    all = np.concatenate((positives,negatives))
    # compute ranks of all positives
    all_ranks = ranks(all,False)[:len(positives)]
    # actual positives
    pos_ranks = ranks(positives,False)
    # (p-1) because ranks start from 1. filtered rank needed for more than one positive sample. For one sample, p = 1, r = a
    filtered_ranks = [a - (p-1) for a,p in itertools.izip(all_ranks,pos_ranks)]
    n = len(negatives) + 1 # adjustment for -1 in quantile function
    # n-r will give the number of negatives after the positive for each positive
    quantiles = np.mean([quantile(r,n) for r in filtered_ranks])
    return np.mean(quantiles)

def get_tuples(batch, volatile=False):
    pairs = []
    for ex in batch:
        pairs.append(ex.pairs)

    return to_var(pairs,volatile=volatile)

def get_labels(batch,volatile=False):
    labels = []
    for ex in batch:
        labels.append(ex.preps)

    return to_var(labels,volatile=volatile)

def to_var(x,volatile=False):
    var = Variable(torch.from_numpy(np.asarray(x)),volatile=volatile)
    if torch.cuda.is_available():
        return var.cuda()
    return var

def pad(arr,pad_size,is_rel=False):
    pad_value = 0
    if pad_size<=len(arr):
        return arr
    for i in range(len(arr),pad_size):
        if is_rel:
            arr.append(pad_value)
        else:
            arr.append([pad_value,pad_value])
    return arr

