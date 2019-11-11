from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from voc_dataset import vocDataset as Dataset
from model import EzDetectNet
from model import EzDetectConfig
from loss import EzDetectLoss

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#training Settings

class opt:
    def __init__(self, batchSize=16, testBatchSize=4, lr=0.001, threads=4, seed=1024, gpu=True):
        self.batchSize = batchSize
        self.testBatchSize = testBatchSize
        self.lr = lr
        self.threads = threads
        self.seed = seed
        self.gpu = gpu

torch.cuda.set_device(2)

print("===========>Loading datasets")
opt = opt()
ezConfig = EzDetectConfig(opt.batchSize, opt.gpu)
train_set = Dataset(ezConfig, True)
test_set = Dataset(ezConfig, False)
train_data_loader = DataLoader(dataset = train_set,
                                num_workers=opt.threads,
                                batch_size=opt.batchSize,
                                shuffle=True)
test_data_loader = DataLoader(dataset = test_set,
                                num_workers=opt.threads,
                                batch_size=opt.batchSize)



print("=======>Building model")
mymodel = EzDetectNet(ezConfig, True)
myloss=  EzDetectLoss(ezConfig)
# optimizer = optim.SGD(mymodel.parameters(), lr = opt.lr,momentum=0.9,weight_decay=1e-4)
optimizer = optim.Adam(mymodel.parameters(), lr = opt.lr)

if ezConfig.gpu == True:
    mymodel.cuda()
    myloss.cuda()

def adjust_learning_rate(optimizer, epoch):
    #每十个epoch，lr下降10倍
    lr = opt.lr * (0.1 ** (epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def doTrain(t):
    mymodel.train()
    for i, batch in enumerate(train_data_loader):
        batchX = batch[0]
        target = batch[1]
        if ezConfig.gpu:
            batchX = batch[0].cuda()
            target = batch[1].cuda()
        x = torch.autograd.Variable(batchX, requires_grad = False)
        confOut, bboxOut = mymodel(x)
        confLoss, bboxLoss = myloss(confOut, bboxOut, target)
        totalLoss = confLoss*4 + bboxLoss

        print(confLoss, bboxLoss)
        print("{} : {} / {} >>>>>>>>>>>>>>>>>>>>>>>>:{}".format(t, i, len(train_data_loader), totalLoss.item()))

        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

def doValidate():
    mymodel.eval()
    lossSum = 0.0
    for i, batch in enumerate(test_data_loader):
        batchX = batch[0]
        target = batch[1]
        if ezConfig.gpu:
            batchX = batch[0].cuda()
            target = batch[1].cuda()
        
        x = torch.autograd.Variable(batchX, requires_grad=False)
        confOut, bboxOut = mymodel(x)

        confLoss, bboxLoss = myloss(confOut, bboxOut, target)
        totalLoss = confLoss * 4 + bboxLoss

        print(confLoss, bboxLoss)
        print("Test : {} / {} >>>>>>>>>>>>>>>>>>>>:{}".format(i, len(test_data_loader), totalLoss.item()))

        lossSum = totalLoss.item() + lossSum
    score = lossSum / len(test_data_loader)
    print("######:{}".format(score))
    return score


####### main function #######
for t in range(50):
    adjust_learning_rate(optimizer, t)
    print("=======>{} epoch:".format(t))
    doTrain(t)
    score = doValidate()
    print(score)
    if (t % 5 ==0):
        torch.save(mymodel.state_dict(),"./model2/model_{}_{}.pth".format(t, str(score)[:4]))

