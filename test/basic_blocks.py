'''
Author: BHM-Bob 2262029386@qq.com
Date: 2023-05-21 00:18:49
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-05-21 12:58:04
Description: Test for Basic Blocks
'''
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

try:
    import bann_flow.bb as bb
except:
    import bb

x = flow.rand([32, 128, 32], device = 'cuda')
net = bb.MultiHeadAttentionLayer(32, 8, 0.3, 'cuda').to('cuda')
print(net(x, x, x).shape, "flow.Size([32, 128, 32])")

net = bb.EncoderLayer(0, 0, 32, 8, 64, 0.3, 'cuda').to('cuda')
print(net(x).shape, "flow.Size([32, 128, 32])")
net = bb.EncoderLayer(0, 0, 32, 4, 64, 0.3, 'cuda', use_FastMHA = False).to('cuda')
print(net(x).shape, "flow.Size([32, 128, 32])")

net = bb.Trans(128, 128, 32, 3, 8, 128, 0.3, 'cuda', bb.EncoderLayer).to('cuda')
print(net(x).shape, "flow.Size([32, 128, 32])")
net = bb.Trans(128, 128, 32, 3, 8, 128, 0.3, 'cuda').to('cuda')
print(net(x).shape, "flow.Size([32, 128])")
net = bb.Trans(128, 128, 32, 3, 8, 128, 0.3, 'cuda', use_enhanced_fc_q = True).to('cuda')
print(net(x).shape, "flow.Size([32, 128])")
net = bb.Trans(128, 32, 32, 3, 8, 128, 0.3, 'cuda', use_enhanced_fc_q = True).to('cuda')
print(net(x).shape, "flow.Size([32, 32])")

x = flow.rand([32, 128, 32], device = 'cuda')
net = bb.OutEncoderLayerAvg(128, 32, 32, 8, 64, 0.3, 'cuda').to('cuda')
print(net(x).shape, "flow.Size([32, 32])")
net = bb.TransAvg(128, 32, 32, 3, 8, 128, 0.3, 'cuda', use_enhanced_fc_q = True).to('cuda')
print(net(x).shape, "oneflow.Size([32, 32])")

loss_fn = nn.MSELoss()
y = flow.rand([8, 32, 32, 32], device = 'cuda')
x = flow.rand([8, 16, 32, 32], device = 'cuda')
net = nn.Conv2d(16, 32, (5, 7), stride=1, padding='same').to('cuda')
optimizer = flow.optim.Adam(net.parameters(), 0.01)
print(net(x).shape, "flow.Size([8, 32, 32, 32])")
optimizer.zero_grad()
loss_fn(net(x), y).backward()
optimizer.step()

x = flow.rand([8, 16, 32, 32], device = 'cuda')
net = bb.ResBlock(bb.CnnCfg(16, 128, stride=2)).to('cuda')
print(net(x).shape, "flow.Size([8, 128, 16, 16])")
net = bb.ResBlockR(bb.CnnCfg(16, 128, stride=2)).to('cuda')
print(net(x).shape, "flow.Size([8, 128, 16, 16])")
net = bb.SABlock(bb.CnnCfg(16, 32)).to('cuda')
print(net(x).shape, "flow.Size([8, 32, 32, 32])")
net = bb.SABlockR(bb.CnnCfg(16, 32)).to('cuda')
print(net(x).shape, "flow.Size([8, 32, 32, 32])")

x = flow.rand([8, 16, 128], device = 'cuda')
net = bb.SABlock1D(bb.CnnCfg(16, 32)).to('cuda')
print(net(x).shape, "flow.Size([8, 32, 128])")
net = bb.SABlock1DR(bb.CnnCfg(16, 32)).to('cuda')
print(net(x).shape, "flow.Size([8, 32, 128])")
