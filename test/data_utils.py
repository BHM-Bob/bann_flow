'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-05-02 20:40:37
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-21 17:42:15
Description: 
'''
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

try:
    from bann_flow.utils import GlobalSettings, Mprint
    from bann_flow.data import DataSetRAM
except:
    from utils import GlobalSettings, Mprint
    from data import DataSetRAM

# global settings
mp = Mprint()
args = GlobalSettings(mp, 'model/')
args.add_arg('read', {'path':r"/home/bhm-bob/文档/AI/AI-Dataset/PTwins/seqs_int16_.seq"})
args.add_arg('load_shape', [64*64])

# load data
# x: [[novel_name:str, seq:tensor], ...]
def bit2tensor(path:str):
    import pickle
    with open(path, 'rb') as f:
        seqs = pickle.load(f)
    for seq in seqs:
        seq[1] = flow.tensor(seq[1], dtype = flow.int32)
    return seqs
ds = DataSetRAM(args, x = args.read['path'],
                x_transfer_origin=bit2tensor,
                x_transfer_gather=lambda x : x[0])
train_loader, test_loader = ds.split([0, 0.7, 1],
                                     x_transformer=[lambda x : x[1][0:args.load_shape[0]],
                                                    lambda x : x[1][0:args.load_shape[0]]])

# training
for x, _ in train_loader:
    print(x.shape)
    break