from os import listdir
from os.path import join
from random import random
from PIL import Image, ImageDraw
import xml.etree.ElementTree

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sampling import sampleEzDetect

__all__ = ["vocClassName", "vocClassID", "vocDataset"] #如果使用“from 模块名 import *”这样的语句来导入模块，程序会导入该模块中所有不以下画线开头的成员（包括变量、函数和类）。
                                                       #__all__ 变量的意义在于为模块定义了一个开放的公共接口。通常来说，只有 __all__ 变量列出的成员，才是希望该模块被外界使用的成员。
                                                       
vocClassName = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

def getVOCInfo(xmlFile):
    root = xml.etree.ElementTree.parse(xmlFile).getroot() #通过读取文件来导入这些数据
    anns = root.findall('object') #找到所有object模块，每个object里都有name,pose,bndbox....

    bboxes = []
    for ann in anns:
        name = ann.find('name').text
        newAnn = {}
        newAnn['category_id'] = name

        bbox = ann.find('bndbox')
        newAnn['bbox'] = [-1,-1,-1,-1]
        newAnn['bbox'][0] = float(bbox.find('xmin').text)
        newAnn['bbox'][1] = float(bbox.find('ymin').text)
        newAnn['bbox'][2] = float(bbox.find('xmax').text)
        newAnn['bbox'][3] = float(bbox.find('ymax').text)
        bboxes.append(newAnn)

    return bboxes

class vocDataset(data.Dataset):
    def __init__(self, config, isTraining=True):
        super(vocDataset, self).__init__()
        self.isTraining = isTraining
        self.config = config
        
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),normalize])

    
    def __getitem__(self, index):
        item = None
        if self.isTraining:
            item = allTrainingData[index % len(allTrainingData)]  #index % len(allTrainingData)= 496 = len(allTestingData)
        else:
            item = allTestingData[index % len(allTestingData)]

        img = Image.open(item[0])

        allBboxes = getVOCInfo(item[1])
        imgWidth, imgHeight = img.size

        targetWidth = int((random() * 0.25 + 0.75) * imgWidth)
        targetHeight = int((random() * 0.25 + 0.75) * imgHeight)

        #对图片进行随机crop
        xmin = int(random() * (imgWidth - targetWidth))
        ymin = int(random() * (imgHeight - targetHeight))
        img = img.crop((xmin, ymin, xmin + targetWidth, ymin + targetHeight))
        img = img.resize((self.config.targetWidth, self.config.targetHeight))
        imgT = self.transforms(img)
        imgT = imgT * 256

        #调整bbox
        bboxes = []
        for i in allBboxes:
            xl = i['bbox'][0] - xmin
            yt = i['bbox'][1] - ymin
            xr = i['bbox'][2] - xmin
            yb = i['bbox'][0] - ymin

            if xl < 0:
                xl = 0
            if xr >= targetWidth:
                xr = targetWidth - 1
            if yt < 0:
                yt = 0
            if yb >= targetHeight:
                yb = targetHeight - 1

            xl = xl / targetWidth
            xr = xr / targetWidth
            yt = yt / targetHeight
            yb = yb / targetHeight

            if xr - xl >= 0.05 and yb - yt >= 0.05:
                bbox = [ vocClassID[ i['category_id']], xl, yt, xr, yb]
                bboxes.append(bbox)

        if len(bboxes)==0 :
            return self[index+1]

        target = sampleEzDetect(self.config, bboxes)
        
        # ###对预测图片进行测试#######
        # draw = ImageDraw.Draw(img)
        # num = int(target[0])
        # for j in range(0, num):
        #     offset = j * 6 #偏移量
        #     if target[offset + 1] < 0:
        #         break
        #     k = int(target[offset + 6])

        #     trueBox = [target[offset + 2],
        #                 target[offset + 3],
        #                 target[offset + 4],
        #                 target[offset + 5]]
        #     predBox = self.config.predBoxes[k]
        # draw.rectangle([trueBox[0]*self.config.targetWidth,
        #             trueBox[1]*self.config.targetHeight,
        #             trueBox[2]*self.config.targetWidth,
        #             trueBox[3]*self.config.targetHeight])

        # draw.rectangle([predBox[0]*self.config.targetWidth,
        #             predBox[1]*self.config.targetHeight,
        #             predBox[2]*self.config.targetWidth,
        #             predBox[3]*self.config.targetHeight, None, "red"])
        # del draw
        # img.save("/tmp/{}.jpg".format(index))

        return imgT, target

    def __len__(self):
        if self.isTraining:
            num = len(allTrainingData) - (len(allTestingData) % self.config.batchSize)
            return num
        else:
            num = len(allTestingData) - (len(allTestingData) % self.config.batchSize)
            return num


#从voc2007中读取数据   
vocClassID = {}

for i in range(len(vocClassName)):
    vocClassID[vocClassName[i]] = i+1
print(vocClassID)
allTrainingData = []
allTestingData = []
#'sjh/pascal/VOCdevkit/VOC2007'
allFloder = ['./VOCdevkit/VOC2007']
# allFloder = ['sjh/pascal/VOCdevkit/VOC2007']
for floder in allFloder:
    imagePath = join(floder, "JPEGImages")
    infoPath = join(floder, "Annotations")
    index = 0
    for f in listdir(imagePath): #遍历9964张原始图片
        if f.endswith('.jpg'):
            imageFile = join(imagePath, f)
            infoFile = join(infoPath, f[:-4] + ".xml")
            if index % 10 == 0: #每十张图片随机抽1个样本做测试
                allTestingData.append((imageFile, infoFile))
            else:
                allTrainingData.append((imageFile, infoFile))
            index = index + 1


            


