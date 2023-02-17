import torch
import torch.nn
import torchvision
import os
import sys
import copy
import PIL.Image

import YOLOv3Net
import YOLOv3_Utils


thresh_objectnessScore = 0.001
thresh_score = 0.01
thresh_iou = 0


anchorBox = (
    ((116 / 416, 90 / 416), (156 / 416, 198 / 416), (373 / 416, 326 / 416)),  # mapA, 归一化后的W x H;
    ((30 / 416, 61 / 416), (62 / 416, 45 / 416), (59 / 416, 119 / 416)),  # mapB, 归一化后的W x H
    ((10 / 416, 13 / 416), (16 / 416, 30 / 416), (33 / 416, 23 / 416))  # mapC, 归一化后的W x H
)  # 在图片大小为416x416时，mapA为13x13，mapB为26x26，mapC为52x52

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


model = YOLOv3Net.YOLOv3Net(20)
modelFolderPath = "./model/"
modelFolderFileList = os.listdir(modelFolderPath)
modelFolderFileList1 = []
for i in modelFolderFileList:
    if os.path.isfile("{}{}".format(modelFolderPath, i)):
        modelFolderFileList1.append(i)
modelFolderFileList = modelFolderFileList1
if len(modelFolderFileList) == 0:
    print("[Fatal Error] Model File Does Not Exist!")
    sys.exit(1)
modelStateDict = torch.load("{}{}".format(modelFolderPath, modelFolderFileList[0]))
model.load_state_dict(modelStateDict)
model.eval()


def calculate_iou(box1, box2, format):
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


def predict(image, yoloImageSize = 416): #image: 原始PIL图像
    #将原始图像转换为YOLOv3所需样式，并转为Tensor
    resizedImage = None
    newImage = PIL.Image.new("RGB", (yoloImageSize, yoloImageSize), (128, 128, 128))
    originCoord = None
    xy_factor = None
    if image.width > image.height:
        resizedImage = image.resize(
            (newImage.width, int(image.height / image.width * newImage.width)),
            PIL.Image.ANTIALIAS)
    else:
        resizedImage = image.resize(
            (int(image.width / image.height * newImage.height), newImage.height),
            PIL.Image.ANTIALIAS)
    xy_factor = resizedImage.width / image.width
    originCoord = (int((newImage.width - resizedImage.width) / 2), int((newImage.height - resizedImage.height) / 2))
    newImage.paste(resizedImage, originCoord)
    pil2TensorFunc = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgTensor = pil2TensorFunc(newImage).to(torch.float32)
    imgTensor = torch.unsqueeze(imgTensor, dim=0)  # 添加一个维度

    global model
    output = model(imgTensor)
    t_output_A = output[0]
    t_output_B = output[1]
    t_output_C = output[2]

    Sigmoid = torch.nn.Sigmoid()
    global anchorBox, VOCClassDict_name2num, VOCClassDict_num2name
    for m in range(0, 3):
        t_output_X = None
        if m == 0:
            t_output_X = t_output_A
        elif m == 1:
            t_output_X = t_output_B
        elif m == 2:
            t_output_X = t_output_C
        gridNum = t_output_X.size()[2]

        t_output_X = torch.squeeze(t_output_X, dim = 0) #C, H, W
        t_output_X = t_output_X.permute(1, 2, 0) #H, W, C
        s = t_output_X.size()
        t_output_X = t_output_X.reshape(s[0], s[1], 3, s[2] // 3) #R, C, B, D
        t_output_X[:, :, :, 0:2] = Sigmoid(t_output_X[:, :, :, 0:2])
        t_output_X[:, :, :, 4:] = Sigmoid(t_output_X[:, :, :, 4:])

        #将s(tx), s(ty)变为yoloImageSize x yoloImageSize坐标系下的x, y坐标
        t_output_X_xy = t_output_X[:, :, :, 0:2].clone()
        t_mask_cx = torch.unsqueeze(torch.linspace(0, gridNum - 1, steps = gridNum), dim = 0).repeat(gridNum, 1)
        t_mask_cy = torch.unsqueeze(torch.linspace(0, gridNum - 1, steps = gridNum), dim = 1).repeat(1, gridNum)
        t_mask_cx = torch.unsqueeze(torch.unsqueeze(t_mask_cx, dim = 2).repeat(1, 1, 3), dim = 3)
        t_mask_cy = torch.unsqueeze(torch.unsqueeze(t_mask_cy, dim = 2).repeat(1, 1, 3), dim = 3)
        t_mask_cxcy = torch.cat((t_mask_cx, t_mask_cy), dim=3)
        t_output_X_xy = torch.add(t_output_X_xy, t_mask_cxcy)
        t_scale = torch.tensor(yoloImageSize / gridNum, dtype=torch.float32)
        t_output_X_xy = torch.mul(t_scale, t_output_X_xy)
        t_output_X[:, :, :, 0:2] = t_output_X_xy

        #将tw, th变为yoloImageSize x yoloImageSize坐标系下的w, h
        t_output_X_wh = t_output_X[:, :, :, 2:4].clone()
        t_output_X_wh = torch.exp(t_output_X_wh)
        t_mask_pwph = torch.tensor([[anchorBox[m][0][0], anchorBox[m][0][1]],
                                    [anchorBox[m][1][0], anchorBox[m][1][1]],
                                    [anchorBox[m][2][0], anchorBox[m][2][1]]],
                                   dtype=torch.float32)
        t_mask_pwph = torch.unsqueeze(torch.unsqueeze(t_mask_pwph, dim = 0), dim = 0).repeat(gridNum, gridNum, 1, 1)
        t_yoloImageSize = torch.tensor(yoloImageSize, dtype=torch.float32)
        t_output_X_wh = torch.mul(torch.mul(t_output_X_wh, t_mask_pwph), t_yoloImageSize)
        t_output_X[:, :, :, 2:4] = t_output_X_wh

        s = t_output_X.size()
        t_output_X = t_output_X.reshape(s[0] * s[1] * s[2], s[3])
        if m == 0:
            t_output_A = t_output_X
        elif m == 1:
            t_output_B = t_output_X
        elif m == 2:
            t_output_C = t_output_X

    output_A = t_output_A.detach().numpy().tolist()
    output_B = t_output_B.detach().numpy().tolist()
    output_C = t_output_C.detach().numpy().tolist()

    global thresh_objectnessScore, thresh_iou, thresh_score
    objBBox = []
    for elm in output_A:
        if elm[4] > thresh_objectnessScore:
            objBBox.append(elm)
    for elm in output_B:
        if elm[4] > thresh_objectnessScore:
            objBBox.append(elm)
    for elm in output_C:
        if elm[4] > thresh_objectnessScore:
            objBBox.append(elm)
    classBBox = []
    for i in range(0, 20):
        classBBox.append([])
    for elm in objBBox:
        classID = 0
        maxClassScore = 0
        for i in range(0, 20):
            if elm[5 + i] > maxClassScore:
                maxClassScore = elm[5 + i]
                classID = i
        score = elm[4] * maxClassScore
        if score > thresh_score:
            tl = copy.deepcopy(elm[0 : 4])
            tl.append(score)
            classBBox[classID].append(tl)
    for i in range(0, 20):
        #冒泡排序
        while True:
            ok = True
            for j in range(0, len(classBBox[i]) - 1):
                if classBBox[i][j + 1][4] > classBBox[i][j][4]:
                    ok = False
                    tmp = classBBox[i][j]
                    classBBox[i][j] = classBBox[i][j + 1]
                    classBBox[i][j + 1] = tmp
            if ok:
                break
        bboxAvailable = []
        for j in range(0, len(classBBox[i])):
            bboxAvailable.append(True)
        for j in range(0, len(classBBox[i]) - 1):
            if bboxAvailable[j]:
                bboxA = (
                    classBBox[i][j][0],
                    classBBox[i][j][1],
                    classBBox[i][j][2],
                    classBBox[i][j][3]
                )
                for k in range(j + 1, len(classBBox[i])):
                    if bboxAvailable[k]:
                        bboxB = (
                            classBBox[i][k][0],
                            classBBox[i][k][1],
                            classBBox[i][k][2],
                            classBBox[i][k][3]
                        )
                        iou = calculate_iou(bboxA, bboxB, 0)
                        if iou > thresh_iou:
                            bboxAvailable[k] = False
        tl = []
        for j in range(0, len(classBBox[i])):
            if bboxAvailable[j]:
                tl.append(classBBox[i][j])
        classBBox[i] = tl

    bboxInfo = []
    for i in range(0, 20):
        for j in range(0, len(classBBox[i])):
            tl = []
            xmin = classBBox[i][j][0] - 0.5 * classBBox[i][j][2]
            ymin = classBBox[i][j][1] - 0.5 * classBBox[i][j][3]
            xmax = classBBox[i][j][0] + 0.5 * classBBox[i][j][2]
            ymax = classBBox[i][j][1] + 0.5 * classBBox[i][j][3]
            tl.append((xmin - originCoord[0]) / xy_factor)
            tl.append((ymin - originCoord[1]) / xy_factor)
            tl.append((xmax - originCoord[0]) / xy_factor)
            tl.append((ymax - originCoord[1]) / xy_factor)
            tl.append(VOCClassDict_num2name[i])
            tl.append(classBBox[i][j][4])
            bboxInfo.append(tl)

    return bboxInfo
