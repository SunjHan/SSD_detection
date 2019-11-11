import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function

from bbox import bboxIOU

__all__ = ["buildPredBoxes", "sampleEzDetect"]

def buildPredBoxes(config):
    predBoxes = []

    for i in range(len(config.mboxes)):
        l = config.mboxes[i][0]
        wid = config.featureSize[l][0] #feature map的宽度
        hei = config.featureSize[l][1]

        wbox = config.mboxes[i][1] #一个像素上产生的bounding box的宽
        hbox = config.mboxes[i][2]

        for y in range(hei):  #对于feature map上的每一个像素
            for x in range(wid):
                xc = (x + 0.5) / wid  #??????????????????????????????????????为什么+0.5，可能是加了一点偏移
                yc = (y + 0.5) / hei
                '''
                xmin = max(0, xc - wbox/2)
                ymin = max(0, yc - hbox/2)
                xmax = min(0, xc + wbox/2)
                ymax = min(0, yc + hbox/2)
                '''
                xmin = xc - wbox/2 #???????????????????????????
                ymin = yc - hbox/2
                xmax = xc + wbox/2
                ymax = yc + hbox/2

                predBoxes.append([xmin, ymin, xmax, ymax])

        return predBoxes

def sampleEzDetect(config, bboxes):
    predBoxes = config.predBoxes

    #preparing ground truth
    truthBoxes=[]
    for i in range(len(bboxes)):
        truthBoxes.append([bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]])

    #computing IOU
    iouMatrix = []
    for i in predBoxes:
        ious = []
        for j in truthBoxes:
            ious.append(bboxIOU(i, j))
        iouMatrix.append(ious)
    iouMatrix = torch.FloatTensor(iouMatrix)
    iouMatrix2 = iouMatrix.clone()

    ii=0
    selectedSamples = torch.FloatTensor(128*1024)


    #首先，对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。
    #positive samples from bi-direction match
    for i in range(len(bboxes)): #对于每个bounding box
        iouViewer = iouMatrix.view(-1)
        iouValues, iouSequence = torch.max(iouViewer, 0) #返回truthBoxes iou最大值和索引,索引是一个值,every for loop choose a max box

        predIndex = iouSequence.item() // len(bboxes) #iouMatirx is predBox * truthBox  #只输出整数
        bboxIndex = iouSequence.item() % len(bboxes)

        if(iouValues.item() > 0.1): #为什么要选大于0.1的
            selectedSamples[ii*6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii*6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii*6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii*6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii*6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii*6 + 6] = predIndex
            ii = ii + 1
        else:
            break #如果剩下之中的最大值都小于0.1则不用再选了

        iouMatrix[:, bboxIndex] = -1  #设置为-1表示选过了？
        iouMatrix[predIndex, :] = -1
        iouMatrix2[predIndex, :] = -1

    #also samples with high iou
    # 对于剩余的未匹配先验框，若某个ground truth的 [公式] 大于某个阈值（一般是0.5），
    # 那么该先验框也与这个ground truth进行匹配。这意味着某个ground truth可能与多个先验框匹配，这是可以的。
    # 但是反过来却不可以，因为一个先验框只能匹配一个ground truth，如果多个ground truth与某个先验框 [公式] 大于阈值，
    # 那么先验框只与IOU最大的那个先验框进行匹配。
    # 参考https://zhuanlan.zhihu.com/p/33544892
    for i in range(len(predBoxes)): #for predBox ，，每一个predbox都要选一个truthbox
        v, _ = iouMatrix2[i].max(0)
        predIndex = i
        bboxIndex = _.item()

        if(v.item() > 0.7): #anchor与真实值的IOU大于0.7
            selectedSamples[ii*6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii*6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii*6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii*6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii*6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii*6 + 6] = predIndex
            ii = ii + 1
            
        elif(v.item() > 0.5):
            selectedSamples[ii*6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii*6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii*6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii*6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii*6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii*6 + 6] = predIndex
            ii = ii + 1
            
    selectedSamples[0] = ii
    return selectedSamples


    