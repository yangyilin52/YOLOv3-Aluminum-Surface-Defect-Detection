import torch
import torch.utils.data
import torchvision.transforms
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import os
import json
import math
import numpy as np

import YOLOv3_DataAugmentations

class AluminumDataset(torch.utils.data.Dataset):
    AluminumClassDict_name2num = {
        "不导电": 0, "擦花": 1, "角位漏底": 2, "桔皮": 3, "漏底": 4,
        "喷流": 5, "漆泡": 6, "起坑": 7, "杂色": 8, "脏点": 9
    }

    AluminumClassDict_num2name = {
        0: "不导电", 1: "擦花", 2: "角位漏底", 3: "桔皮", 4: "漏底",
        5: "喷流", 6: "漆泡", 7: "起坑", 8: "杂色", 9: "脏点"
    }

    anchorBox = ( #Todo: 需要更改
        ((116 / 416, 90 / 416), (156 / 416, 198 / 416), (373 / 416, 326 / 416)),  # mapA, 归一化后的W x H;
        ((30 / 416, 61 / 416), (62 / 416, 45 / 416), (59 / 416, 119 / 416)),  # mapB, 归一化后的W x H
        ((10 / 416, 13 / 416), (16 / 416, 30 / 416), (33 / 416, 23 / 416))  # mapC, 归一化后的W x H
    ) #在图片大小为416x416时，mapA为13x13，mapB为26x26，mapC为52x52

    def __init__(self, path, yoloImageSize = 416):
        super().__init__()
        self.datasetPath = path
        if self.datasetPath[-1] != "/":
            self.datasetPath = self.datasetPath + "/"

        self.imgFileExtension = ".jpg"
        self.antFileExtension = ".json"

        self.fileNameList = []
        allFileList = os.listdir(self.datasetPath)
        for i in allFileList:
            if os.path.isfile("{}{}".format(self.datasetPath, i)):
                if i[-4:] == ".jpg":
                    self.fileNameList.append(i[0:-4])

        self.yoloImageSize = yoloImageSize

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Loading Aluminum Dataset for YOLO v3.")
        print("This Dataset Contains {} Images and Annotations.".format(len(self.fileNameList)))
        print("")

    def __getitem__(self, index):
        imgFilePath = "{}{}{}".format(self.datasetPath, self.fileNameList[index], self.imgFileExtension)
        antFilePath = "{}{}{}".format(self.datasetPath, self.fileNameList[index], self.antFileExtension)

        #原始图片
        originalImage = PIL.Image.open(imgFilePath) # pillow加载图片时默认的数据格式为RGB
        #原始标签信息
        fp_antFile = open(file = antFilePath, mode = "r", encoding = "UTF-8")
        jsonData = json.load(fp_antFile)
        fp_antFile.close()
        jsonData_BBoxes = jsonData["shapes"]
        antInfoList_original = []  # dim2: 0 xmin, 1 ymin, 2 xmax, 3 ymax, 4 name. In original size image.
        for i in jsonData_BBoxes:
            tl = []
            tl.append(i["points"][0][0]) #xmin
            tl.append(i["points"][0][1]) #ymin
            tl.append(i["points"][2][0] - 1) #xmax
            tl.append(i["points"][2][1] - 1) #ymax
            tl.append(i["label"]) #classname
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
            tl.append(AluminumDataset.AluminumClassDict_name2num[ant[4]])
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
                box_a = (0, 0, AluminumDataset.anchorBox[i // 3][i % 3][0] * self.yoloImageSize, AluminumDataset.anchorBox[i // 3][i % 3][1] * self.yoloImageSize)
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
                    sLabelData_X[row][col][boxID][2] = math.log(y[2] / (AluminumDataset.anchorBox[anchorID // 3][anchorID % 3][0] * self.yoloImageSize))
                    sLabelData_X[row][col][boxID][3] = math.log(y[3] / (AluminumDataset.anchorBox[anchorID // 3][anchorID % 3][1] * self.yoloImageSize))
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
            在图上画出边界框并标注类别名称，用于Debug。
            该函数会在一个新的PIL图片对象上绘图，不会影响原来的PIL图片对象。
        """
        img = img.copy()
        draw = PIL.ImageDraw.Draw(img)
        font_consola = PIL.ImageFont.truetype("./fonts/msyh.ttc", 18)
        for b in bboxInfo:
            draw.rectangle(xy=(b[0], b[1], b[2], b[3]), fill=None, outline=color, width=1)
            draw.text(xy=(b[0], b[1] - 24), text=b[4], fill=color, font=font_consola)
        return img
