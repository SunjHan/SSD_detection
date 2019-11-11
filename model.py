 
import os
import math
import torch
import torch.nn as nn
import torch.autograd as Function
import torch.nn.functional as F
import torchvision.models as models

from sampling import buildPredBoxes

__all__ = ['EzDetectConfig','EzDetectNet', 'ReorgModule']

class EzDetectConfig(object):  #设置了每一个产生的bounding box的大小
    def __init__(self, batchSize = 4, gpu = False):
        super(EzDetectConfig, self).__init__()
        self.batchSize = batchSize
        self.gpu = gpu
        self.classNumber = 21
        self.targetWidth = 330
        self.targetHeight = 330
        self.featureSize = [[42, 42],
                            [21, 21],
                            [11, 11],
                            [6, 6],
                            [3, 3]]
        #[min ,max, ratio]
        priorConfig = [[0.10, 0.25, 2],
                        [0.25, 0.40, 2, 3],
                        [0.40, 0.55, 2, 3],
                        [0.55, 0.70, 2, 3],
                        [0.70, 0.85, 2]]
        self.mboxes = []
        for i in range(len(priorConfig)): #len = 5
            minSize = priorConfig[i][0]
            maxSize = priorConfig[i][1]
            meanSize = math.sqrt(minSize * maxSize)
            ratios = priorConfig[i][2:]

            #aspect ratio 1 fpr min and max
            self.mboxes.append([i, minSize, minSize])  #prior box的最小和最大边长 #加入两个框，一个最小一个最大
            self.mboxes.append([i, meanSize, meanSize])
        
            #other aspect ratio
            for r in ratios:   #加入2*ratios个框
                ar = math.sqrt(r)
                self.mboxes.append([i, minSize*ar, minSize/ar]) 
                self.mboxes.append([i, minSize/ar, minSize*ar])
        self.predBoxes = buildPredBoxes(self)



class EzDetectNet(nn.Module):
    def __init__(self, config, pretrained=False):
        super(EzDetectNet, self).__init__()
        self.config = config
        resnet = models.resnet50(pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layer5 = nn.Sequential(
            #直到第五层才开始自定义
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        inChannles = [512, 1024, 2048, 1024, 512]
        self.locConvs = []
        self.confConvs = []
        for i in range(len(config.mboxes)):
            inSize = inChannles[config.mboxes[i][0]]
            confConv = nn.Conv2d(inSize, config.classNumber, kernel_size=3, stride=1, padding=1, bias=True)
            locConv = nn.Conv2d(inSize, 4, kernel_size=3, stride=1, padding=1, bias=True)
            self.locConvs.append(locConv)
            self.confConvs.append(confConv)

            super(EzDetectNet, self).add_module("{}_conf".format(i), confConv)
            super(EzDetectNet, self).add_module("{}_loc".format(i), locConv)
        
    def forward(self, x):
        batchSize = x.size()[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)

        featureSource = [l2, l3, l4, l5, l6]

        confs = []
        locs = []
        for i in range(len(self.config.mboxes)):
            x = featureSource[self.config.mboxes[i][0]]
            loc = self.locConvs[i](x)
            loc = loc.permute(0, 2, 3, 1) #将tensor的维度换位
            loc = loc.contiguous() #当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样。
            loc = loc.view(batchSize, -1, 4) #view():改变tensor的形状
            locs.append(loc)

            conf = self.confConvs[i](x)
            conf = conf.permute(0, 2, 3, 1)  #[batch_num, channel, height, width]-->[batch_num, height, width, channel]
            conf = conf.contiguous()
            conf = conf.view(batchSize, -1, self.config.classNumber)
            confs.append(conf)
        
        locResult = torch.cat(locs, 1) #concatnate
        confResult = torch.cat(confs, 1)

        return confResult, locResult