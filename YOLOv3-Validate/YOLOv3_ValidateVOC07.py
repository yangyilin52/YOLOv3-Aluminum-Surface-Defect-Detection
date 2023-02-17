import YOLOv3_CorePredict
import YOLOv3_Utils
import PIL.Image
import os
import xml.dom.minidom
import random
import cv2
import numpy


#可调参数
datasetPath = "C:/Users/Admin/Desktop/VOC2007数据集/VOCtest_06-Nov-2007/"
#datasetPath = "C:/Users/Admin/Desktop/VOC2007数据集/VOCtrainval_06-Nov-2007/"


class SimpleDatasetReader_VOC07:
    VOC07ClassDict_num2name = {
        0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
        5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
        10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
        15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"
    }

    VOC07ClassDict_name2num = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
        "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9,
        "diningtable": 10, "dog": 11, "horse": 12, "motorbike": 13, "person": 14,
        "pottedplant": 15, "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19
    }

    def __init__(self, path):
        self.datasetPath = path
        self.currentIndex = 0
        if self.datasetPath[-1] != "/":
            self.datasetPath = self.datasetPath + "/"

        self.imgFolderPath = "{}VOCdevkit/VOC2007/JPEGImages/".format(self.datasetPath)
        self.imgFileList = os.listdir(self.imgFolderPath)
        imgFileList1 = []
        for i in self.imgFileList:
            if os.path.isfile("{}{}".format(self.imgFolderPath, i)):
                imgFileList1.append(i)
        self.imgFileList = imgFileList1

        self.antFolderPath = "{}VOCdevkit/VOC2007/Annotations/".format(self.datasetPath)
        self.antFileList = os.listdir(self.antFolderPath)
        antFileList1 = []
        for i in self.antFileList:
            if os.path.isfile("{}{}".format(self.antFolderPath, i)):
                antFileList1.append(i)
        self.antFileList = antFileList1

    def getImageData(self):
        #获取PIL图片
        imgFilePath = "{}{}".format(self.imgFolderPath, self.imgFileList[self.currentIndex])
        pillowImg = PIL.Image.open(imgFilePath)  # pillow加载图片时默认的数据格式为RGB

        #获取标签信息
        antFilePath = "{}{}".format(self.antFolderPath, self.antFileList[self.currentIndex])
        domTree = xml.dom.minidom.parse(antFilePath)
        rootNode = domTree.documentElement
        objectNodes = rootNode.getElementsByTagName("object")
        antInfoList = []  # dim2:0 name, 1 xmin, 2 ymin, 3 xmax, 4 ymax, 5 imgWidth, 6 imgHeight
        for i in objectNodes:
            tl = []
            tl.append(int(i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")[0].childNodes[0].data))
            tl.append(int(i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")[0].childNodes[0].data))
            tl.append(int(i.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")[0].childNodes[0].data))
            tl.append(int(i.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")[0].childNodes[0].data))
            tl.append(i.getElementsByTagName("name")[0].childNodes[0].data)
            antInfoList.append(tl)

        return pillowImg, imgFilePath, antInfoList, antFilePath

    def randomNext(self):
        self.currentIndex = random.randint(0, len(self.imgFileList) - 1)

    def next(self):
        self.currentIndex += 1
        if self.currentIndex == len(self.imgFileList):
            self.currentIndex = 0

    def previous(self):
        self.currentIndex -= 1
        if self.currentIndex < 0:
            self.currentIndex = len(self.imgFileList) - 1

    def goTo(self, index):
        if  (index >= 0) and (index < len(self.imgFileList)):
            self.currentIndex = index

print("在VOC2007数据集上验证Yolo V3\n")
voc07Reader = SimpleDatasetReader_VOC07(datasetPath)

inputVal = ""
while True:
    print("请选择验证的模式:\n  1.自动顺序验证\n  2.自动随机验证\n  3.手动验证\n  4.指定图片验证")
    inputVal = input(">>> ")
    if inputVal == "1" or inputVal == "2" or inputVal == "3" or inputVal == "4":
        break

if inputVal == "1":
    print("[自动顺序验证]")
    while True:
        data = voc07Reader.getImageData()
        pillowImage = data[0]
        imageFilePath = data[1]
        antInfoList = data[2]
        antFilePath = data[3]
        result = YOLOv3_CorePredict.predict(pillowImage)
        #YOLOv3_Utils.drawBBox(pillowImage, 0, antInfoList, (255, 0, 255))
        YOLOv3_Utils.drawBBox(pillowImage, 1, result, (0, 255, 0))
        cvImage = cv2.cvtColor(numpy.asarray(pillowImage), cv2.COLOR_RGB2BGR)
        windowsName = "Result"
        print("--------------------------------------")
        print("图像文件: {}\n标注文件: {}".format(imageFilePath, antFilePath))
        cv2.imshow(windowsName, cvImage)
        cv2.waitKey(5000)
        voc07Reader.next()
elif inputVal == "2":
    print("[自动随机验证]")
    voc07Reader.randomNext()
    while True:
        data = voc07Reader.getImageData()
        pillowImage = data[0]
        imageFilePath = data[1]
        antInfoList = data[2]
        antFilePath = data[3]
        result = YOLOv3_CorePredict.predict(pillowImage)
        YOLOv3_Utils.drawBBox(pillowImage, 0, antInfoList, (255, 0, 255))
        YOLOv3_Utils.drawBBox(pillowImage, 1, result, (0, 255, 0))
        cvImage = cv2.cvtColor(numpy.asarray(pillowImage), cv2.COLOR_RGB2BGR)
        windowsName = "Result"
        print("--------------------------------------")
        print("图像文件: {}\n标注文件: {}".format(imageFilePath, antFilePath))
        cv2.imshow(windowsName, cvImage)
        cv2.waitKey(5000)
        voc07Reader.randomNext()
elif inputVal == "3":
    print("[手动验证]\n操作方法:\nd -> 顺序下一张\na -> 顺序上一张\nr -> 随机下一张")
    while True:
        data = voc07Reader.getImageData()
        pillowImage = data[0]
        imageFilePath = data[1]
        antInfoList = data[2]
        antFilePath = data[3]
        result = YOLOv3_CorePredict.predict(pillowImage)
        YOLOv3_Utils.drawBBox(pillowImage, 0, antInfoList, (255, 0, 255))
        YOLOv3_Utils.drawBBox(pillowImage, 1, result, (0, 255, 0))
        cvImage = cv2.cvtColor(numpy.asarray(pillowImage), cv2.COLOR_RGB2BGR)
        windowsName = "Result"
        print("--------------------------------------")
        print("图像文件: {}\n标注文件: {}\n".format(imageFilePath, antFilePath))
        cv2.imshow(windowsName, cvImage)
        while True:
            key = cv2.waitKey()
            if key == 100: #Pressed 'd'
                voc07Reader.next()
                print("顺序下一张")
                break
            elif key == 97: #Pressed 'a'
                voc07Reader.previous()
                print("顺序上一张")
                break
            elif key == 114: #Pressed 'r'
                voc07Reader.randomNext()
                print("随机下一张")
                break
elif inputVal == "4":
    print("[指定图片验证]")
    while True:
        print("请输入图片序号:")
        inputVal = input(">>> ")
        try:
            picIndex = int(inputVal)
        except:
            print("Error: 输入值不是序号")
        else:
            voc07Reader.goTo(picIndex)
            data = voc07Reader.getImageData()
            pillowImage = data[0]
            imageFilePath = data[1]
            antInfoList = data[2]
            antFilePath = data[3]
            result = YOLOv3_CorePredict.predict(pillowImage)
            YOLOv3_Utils.drawBBox(pillowImage, 0, antInfoList, (255, 0, 255))
            YOLOv3_Utils.drawBBox(pillowImage, 1, result, (0, 255, 0))
            cvImage = cv2.cvtColor(numpy.asarray(pillowImage), cv2.COLOR_RGB2BGR)
            windowsName = "Result"
            print("图像文件: {}\n标注文件: {}\n".format(imageFilePath, antFilePath))
            cv2.imshow(windowsName, cvImage)
            cv2.waitKey()
            cv2.destroyWindow(windowsName)
