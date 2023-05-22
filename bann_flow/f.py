'''
Date: 2023-05-21 18:08:06
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-21 18:15:14
FilePath: /bann_flow/bann_flow/f.py
Description: 
'''

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

def diag_embed(arr: flow.Tensor):
    assert len(arr.shape) == 1
    diag = flow.zeros((arr.shape[0], arr.shape[0]), dtype = arr.dtype, device=arr.device)
    idxs = flow.arange(arr.shape[0])
    diag[idxs, idxs] = arr
    return diag