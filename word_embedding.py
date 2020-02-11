import torch
import numpy as np

__vec = []
vec_len = 300

with open('data/dict.txt', 'r') as f:
    for line in f.readlines():
        vec = list(map(float, line.split(' ')[1:]))
        if len(vec) > 1:
            __vec.append(vec)
        else:
            __vec.append([])

def get_vec(idx):
    try:
        return __vec[idx][:vec_len]
    except:
        return [0]