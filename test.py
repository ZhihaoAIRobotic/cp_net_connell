import torch
from cpnet2 import CPNET
from Cornell import Cornell
import torch.optim as optim
import datetime
import os
import argparse
import logging
import tensorboardX
import cv2
import numpy as np

net = CPNET()

net.load_state_dict(torch.load('model_test/model2.pkl'))
device = torch.device("cuda:0")
net = net.to(device)
Path_dataset = '../cornell_data'

# 首先初始化一个tensorboardX
tb = tensorboardX.SummaryWriter('log_tb')  # 括号里为数据存放的地址

# train dataset
train_dataset = Cornell(Path_dataset, start=0.0, end=0.9)
x,y,_,_,_=train_dataset.__getitem__(174)
x = x.unsqueeze(1)
y = y.unsqueeze(1)
xc = x.to(device)
yc = y.to(device)

lossd = net.compute_loss(xc, yc)
im=lossd['pred']['pos'][0].detach().cpu().numpy()
im=im.reshape((300,300))

print(np.argmax(im))
print(im.shape)
xx=np.zeros((300,300))
xx[114][160]=255
cv2.imshow('1',xx)
cv2.waitKey(0)
