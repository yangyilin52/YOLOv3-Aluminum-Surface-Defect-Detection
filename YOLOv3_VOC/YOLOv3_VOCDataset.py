import torch
import torch.utils.data
import torchvision.transforms
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import os
import xml.dom.minidom
import math
import random
import numpy as np

import YOLOv3_DataAugmentations

class VOCDataset(torch.utils.data.Dataset):
    VOCClassDict_name2num = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
        "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
        "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
        "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19
    }

    VOCClassDict_num2name = {
        0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
        5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
        10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
        15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"
    }

    anchorBox = (
        ((116 / 416, 90 / 416), (156 / 416, 198 / 416), (373 / 416, 326 / 416)),  # mapA, 归一化后的W x H;
        ((30 / 416, 61 / 416), (62 / 416, 45 / 416), (59 / 416, 119 / 416)),  # mapB, 归一化后的W x H
        ((10 / 416, 13 / 416), (16 / 416, 30 / 416), (33 / 416, 23 / 416))  # mapC, 归一化后的W x H
    ) #在图片大小为416x416时，mapA为13x13，mapB为26x26，mapC为52x52

    def __init__(self, path, yoloImageSize = 416, useDataAugmentation = True):
        super().__init__()
        datasetPath = path
        if datasetPath[-1] != "/":
            datasetPath = datasetPath + "/"

        devkitPath = "{}VOCdevkit/".format(datasetPath)
        year = int(os.listdir(devkitPath)[0][3:])

        self.imgFolderPath = "{}VOCdevkit/VOC{}/JPEGImages/".format(datasetPath, year)
        self.imgFileExtension = ".jpg"
        self.antFolderPath = "{}VOCdevkit/VOC{}/Annotations/".format(datasetPath, year)
        self.antFileExtension = ".xml"

        self.fileNameList = []
        imgFileList = os.listdir(self.imgFolderPath)
        for i in imgFileList:
            if os.path.isfile("{}{}".format(self.imgFolderPath, i)):
                self.fileNameList.append(i[0:-4])

        self.yoloImageSize = yoloImageSize
        self.useDataAugmentation = useDataAugmentation

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Loading VOC{} Dataset for YOLO v3.".format(year))
        print("This Dataset Contains {} Images and Annotations.".format(len(self.fileNameList)))
        print("")

    def __getitem__(self, index):
        imgFilePath = "{}{}{}".format(self.imgFolderPath, self.fileNameList[index], self.imgFileExtension)
        antFilePath = "{}{}{}".format(self.antFolderPath, self.fileNameList[index], self.antFileExtension)

        #原始图片
        originalImage = PIL.Image.open(imgFilePath) # pillow加载图片时默认的数据格式为RGB
        #原始标签信息
        domTree = xml.dom.minidom.parse(antFilePath)
        rootNode = domTree.documentElement
        objectNodes = rootNode.getElementsByTagName("object")
        antInfoList_original = []  # dim2: 0 xmin, 1 ymin, 2 xmax, 3 ymax, 4 name. In original size image.
        for i in objectNodes:
            tl = []
            tl.append(int(float(i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")[0].childNodes[0].data)))
            tl.append(int(float(i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")[0].childNodes[0].data)))
            tl.append(int(float(i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")[0].childNodes[0].data)))
            tl.append(int(float(i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")[0].childNodes[0].data)))
            tl.append(i.getElementsByTagName("name")[0].childNodes[0].data)
            antInfoList_original.append(tl)

        #缩放后的图片
        resizedImage = None
        if originalImage.width > originalImage.height:
            resizedImage = originalImage.resize((self.yoloImageSize, int(originalImage.height / originalImage.width * self.yoloImageSize)),
                                                PIL.Image.ANTIALIAS)
        else:
            resizedImage = originalImage.resize((int(originalImage.width / originalImage.height * self.yoloImageSize), self.yoloImageSize),
                                                PIL.Image.ANTIALIAS)
        xy_factor_0 = resizedImage.width / originalImage.width
        #缩放后的标签信息
        antInfoList_resized = [] # dim2: 0 xmin, 1 ymin, 2 xmax, 3 ymax, 4 name. In resized image.
        for ant in antInfoList_original:
            tl = []
            tl.append(int(ant[0] * xy_factor_0))
            tl.append(int(ant[1] * xy_factor_0))
            tl.append(int(ant[2] * xy_factor_0))
            tl.append(int(ant[3] * xy_factor_0))
            tl.append(ant[4])
            antInfoList_resized.append(tl)

        finalImage = PIL.Image.new("RGB", (self.yoloImageSize, self.yoloImageSize), (128, 128, 128))
        antInfoList_final = [] # dim2: 0 xmin, 1 ymin, 2 xmax, 3 ymax, 4 name. In (yoloImageSize x yoloImageSize) Image.

        if self.useDataAugmentation:
            StartAugmentation = YOLOv3_DataAugmentations.StartAugmentation()
            EndAugmentation = YOLOv3_DataAugmentations.EndAugmentation()
            RandomBrightness = YOLOv3_DataAugmentations.RandomBrightness(32)
            RandomContrast = YOLOv3_DataAugmentations.RandomContrast(0.5, 1.5)
            RandomSaturation = YOLOv3_DataAugmentations.RandomSaturation(0.5, 1.5)
            RandomHue = YOLOv3_DataAugmentations.RandomHue(9)
            RandomLightingNoise = YOLOv3_DataAugmentations.RandomLightingNoise()
            RandomMirror = YOLOv3_DataAugmentations.RandomMirror()
            RandomResize = YOLOv3_DataAugmentations.RandomResize(0.8, 1.0, 0.8, 1.0, 25)

            augImage, antInfoList_aug = StartAugmentation(resizedImage, antInfoList_resized)
            augImage, antInfoList_aug = RandomBrightness(augImage, antInfoList_aug)
            augImage, antInfoList_aug = RandomContrast(augImage, antInfoList_aug)
            augImage, antInfoList_aug = RandomSaturation(augImage, antInfoList_aug)
            augImage, antInfoList_aug = RandomHue(augImage, antInfoList_aug)
            augImage, antInfoList_aug = RandomLightingNoise(augImage, antInfoList_aug)
            augImage, antInfoList_aug = RandomMirror(augImage, antInfoList_aug)
            augImage, antInfoList_aug = RandomResize(augImage, antInfoList_aug)
            augImage, antInfoList_aug = EndAugmentation(augImage, antInfoList_aug)

            originCoord = (
                random.randint(0, finalImage.width - augImage.width),
                random.randint(0, finalImage.height - augImage.height)
            )
            finalImage.paste(augImage, originCoord)
            for ant in antInfoList_aug:
                tl = []
                tl.append(ant[0] + originCoord[0])
                tl.append(ant[1] + originCoord[1])
                tl.append(ant[2] + originCoord[0])
                tl.append(ant[3] + originCoord[1])
                tl.append(ant[4])
                antInfoList_final.append(tl)
        else:
            originCoord = (
                int((finalImage.width - resizedImage.width) / 2),
                int((finalImage.height - resizedImage.height) / 2)
            )
            finalImage.paste(resizedImage, originCoord)
            for ant in antInfoList_resized:
                tl = []
                tl.append(ant[0] + originCoord[0])
                tl.append(ant[1] + originCoord[1])
                tl.append(ant[2] + originCoord[0])
                tl.append(ant[3] + originCoord[1])
                tl.append(ant[4])
                antInfoList_final.append(tl)

        yolov3InfoList = [] # dim2: 0 centerX, 1 centerY, 2 width, 3 height, 4 nameID. In (yoloImageSize x yoloImageSize) Image.
        for ant in antInfoList_final:
            tl = []
            tl.append((ant[0] + ant[2]) / 2)
            tl.append((ant[1] + ant[3]) / 2)
            tl.append(ant[2] - ant[0])
            tl.append(ant[3] - ant[1])
            tl.append(VOCDataset.VOCClassDict_name2num[ant[4]])
            yolov3InfoList.append(tl)

        # 初始化labelData_X，每一个预测边界框的标签数据的格式为 s(tx), s(ty), tw, th, s(to), s(tc1), ... , s(tc20)
        # 其中s(x) = sigmoid(x)
        labelData_A = np.zeros((self.yoloImageSize // 32, self.yoloImageSize // 32, 3, 25)).tolist()

        labelData_B = np.zeros((self.yoloImageSize // 16, self.yoloImageSize // 16, 3, 25)).tolist()

        labelData_C = np.zeros((self.yoloImageSize // 8, self.yoloImageSize // 8, 3, 25)).tolist()

        # 初始化labelTag_X
        # 每一个预测边界框的标记默认是”不负责预测真实边界框“。如果在labelTag_X里被特殊标记了，那么: 1 ”忽略“，2 ”负责预测某个真实边界框“
        # 本程序假定每张图最多有100个真实边界框
        labelTag_A = []

        labelTag_B = []

        labelTag_C = []

        for y in yolov3InfoList:
            iouList = []
            anchorIDList = []
            box_t = (0, 0, y[2], y[3])
            for i in range(0, 9):
                box_a = (0, 0, VOCDataset.anchorBox[i // 3][i % 3][0] * self.yoloImageSize, VOCDataset.anchorBox[i // 3][i % 3][1] * self.yoloImageSize)
                iouList.append(self.calculate_iou(box_a, box_t, 0))
                anchorIDList.append(i)
            #冒泡排序
            while True:
                ok = True
                for i in range(0, len(iouList) - 1):
                    if iouList[i + 1] > iouList[i]:
                        ok = False
                        tmp = iouList[i]
                        iouList[i] = iouList[i + 1]
                        iouList[i + 1] = tmp
                        tmp = anchorIDList[i]
                        anchorIDList[i] = anchorIDList[i + 1]
                        anchorIDList[i + 1] = tmp
                if ok:
                    break

            for i in range(0, 9):
                anchorID = anchorIDList[i]
                sLabelData_X = labelData_A
                sLabelTag_X = labelTag_A
                if (anchorID // 3) == 0:
                    sLabelData_X = labelData_A
                    sLabelTag_X = labelTag_A
                elif (anchorID // 3) == 1:
                    sLabelData_X = labelData_B
                    sLabelTag_X = labelTag_B
                elif (anchorID // 3) == 2:
                    sLabelData_X = labelData_C
                    sLabelTag_X = labelTag_C
                gridNum = len(sLabelData_X)
                row = int(y[1] / (self.yoloImageSize / gridNum))
                col = int(y[0] / (self.yoloImageSize / gridNum))
                boxID = anchorID % 3
                tag = [row, col, boxID, 2] #”负责预测某个真实边界框“
                if tag not in sLabelTag_X:
                    sLabelTag_X.append(tag)
                    sLabelData_X[row][col][boxID][0] = y[0] / (self.yoloImageSize / gridNum) - col
                    sLabelData_X[row][col][boxID][1] = y[1] / (self.yoloImageSize / gridNum) - row
                    sLabelData_X[row][col][boxID][2] = math.log(y[2] / (VOCDataset.anchorBox[anchorID // 3][anchorID % 3][0] * self.yoloImageSize))
                    sLabelData_X[row][col][boxID][3] = math.log(y[3] / (VOCDataset.anchorBox[anchorID // 3][anchorID % 3][1] * self.yoloImageSize))
                    sLabelData_X[row][col][boxID][4] = 1
                    sLabelData_X[row][col][boxID][5 + y[4]] = 1
                    break

        for i in range(len(labelTag_A), 3 * 100):
            labelTag_A.append([-1, -1, -1, -1])
        for i in range(len(labelTag_B), 3 * 100):
            labelTag_B.append([-1, -1, -1, -1])
        for i in range(len(labelTag_C), 3 * 100):
            labelTag_C.append([-1, -1, -1, -1])

        imgTensor = self.transform(finalImage).to(torch.float32)

        t_labelData_A = torch.tensor(labelData_A, dtype = torch.float32)
        t_labelData_B = torch.tensor(labelData_B, dtype = torch.float32)
        t_labelData_C = torch.tensor(labelData_C, dtype=torch.float32)
        t_labelTag_A = torch.tensor(labelTag_A, dtype=torch.int16)
        t_labelTag_B = torch.tensor(labelTag_B, dtype=torch.int16)
        t_labelTag_C = torch.tensor(labelTag_C, dtype=torch.int16)

        return imgTensor, t_labelData_A, t_labelData_B, t_labelData_C, t_labelTag_A, t_labelTag_B, t_labelTag_C

    def __len__(self):
        return len(self.fileNameList)

    def calculate_iou(self, box1, box2, format):
        # format: 0, 1。
        # 当format == 0，box1和box2格式应为(center_x, center_y, width, height)
        # 当format == 1，box1和box2的格式应为(x1, y1, x2, y2)

        # 计算box1和box2的iou
        if format == 0:
            box1 = (box1[0] - 0.5 * box1[2], box1[1] - 0.5 * box1[3], box1[0] + 0.5 * box1[2], box1[1] + 0.5 * box1[3])
            box2 = (box2[0] - 0.5 * box2[2], box2[1] - 0.5 * box2[3], box2[0] + 0.5 * box2[2], box2[1] + 0.5 * box2[3])

        intersect_box = [0., 0., 0., 0.]  # box1和box2的交集
        if box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3]:
            pass
        else:
            intersect_box[0] = max(box1[0], box2[0])
            intersect_box[1] = max(box1[1], box2[1])
            intersect_box[2] = min(box1[2], box2[2])
            intersect_box[3] = min(box1[3], box2[3])

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  # box1面积
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])  # box2面积
        area_intersect = (intersect_box[2] - intersect_box[0]) * (intersect_box[3] - intersect_box[1])  # 交集面积

        if area_intersect > 0:
            return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
        else:
            return 0

    def drawBBox(self, img, bboxInfo, color):
        """
        参数:
            img: PIL图片对象
            bboxInfo: 边界框信息列表，结构为[[xmin, ymin, xmax, ymax, classname], ...]
            color: 格式为(R, G, B)
        功能:
            在图上画出边界框并标注边界框内物体类别名称，用于Debug。
            该函数会在一个新的PIL图片对象上绘图，不会影响原来的PIL图片对象。
        """
        img = img.copy()
        draw = PIL.ImageDraw.Draw(img)
        myFont = PIL.ImageFont.truetype("./Fonts/msyhbd.ttc", 20)
        for b in bboxInfo:
            draw.rectangle(xy=(b[0], b[1], b[2], b[3]), fill=None, outline=color, width=1)
            draw.text(xy=(b[0], b[1] - 20), text=b[4], fill=color, font=myFont)
        return img
