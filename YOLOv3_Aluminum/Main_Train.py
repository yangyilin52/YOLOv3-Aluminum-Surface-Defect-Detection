import torch
import torch.utils.data
import os

import YOLOv3_AluminumDataset
import YOLOv3Net
import YOLOv3_Loss
import Logger


#可调参数
useGPU = True
batchSize = 16
learningRate = 1e-4
epochAmount = 20
lossScale = 1.0
#------------------------------
continueTraining = False
ct_modelFilePath = ""
ct_initialEpochNum = -1
#------------------------------
outputPath = "./Output/a0/"


def transferStateDict(pretrainedStateDict, modelStateDict):
    for k, v in pretrainedStateDict.items():
        if k in modelStateDict.keys():
            modelStateDict[k] = v


if outputPath[-1] != "/":
    outputPath = outputPath + "/"
try:
    os.mkdir(outputPath)
except:
    pass
Logger.setPassedTimeFormat2hms()
Logger.setLogFile(True, "{}Log.txt".format(outputPath))
#记录重要训练信息
Logger.log("[Program Parameters]")
Logger.log("useGPU = {}".format(useGPU))
Logger.log("batchSize = {}".format(batchSize))
Logger.log("learningRate = {}".format(learningRate))
Logger.log("lossScale = {}".format(lossScale))
Logger.log("")


model = YOLOv3Net.YOLOv3Net(10)
if continueTraining:
    model_stateDict = torch.load(ct_modelFilePath)
    model.load_state_dict(model_stateDict)
else:
    darknet53_stateDict = torch.load("./darknet53_75.42.pth")
    model_stateDict = model.state_dict()
    transferStateDict(darknet53_stateDict, model_stateDict)
    model.load_state_dict(model_stateDict)


trainDataset = YOLOv3_AluminumDataset.AluminumDataset("C:/Users/Admin/Desktop/VOC2007数据集/VOCtrainval_06-Nov-2007/", 416, True)
trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
validateDataset = YOLOv3_AluminumDataset.AluminumDataset("C:/Users/Admin/Desktop/VOC2007数据集/VOCtest_part/", 416, False)
validateLoader = torch.utils.data.DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)


rangeObj = None
if continueTraining:
    rangeObj = range(ct_initialEpochNum, ct_initialEpochNum + epochAmount)
else:
    rangeObj = range(1, 1 + epochAmount)
if useGPU:
    Logger.log("Compute Device: Nvidia GPU")
else:
    Logger.log("Compute Device: CPU")
criterion = YOLOv3_Loss.YOLOv3Loss(useGPU, lossScale)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9, weight_decay=5.0e-4)  #SGD优化器
for epoch in rangeObj:
    Logger.log("************************* Epoch {} *************************".format(epoch))
    if useGPU:
        model.cuda()

    #Train部分
    Logger.log("Training...")
    model.train()
    epochTrainLoss = 0.0
    for i, data in enumerate(trainLoader, 1):
        Logger.log("[ Epoch:", epoch, "| Batch:", i, "]")
        img = data[0]
        label = data[1:]
        if useGPU:
            img = img.cuda()
            for j in range(0, 3):
                label[j] = label[j].cuda()
        output = model(img)
        loss = criterion(output, label, YOLOv3_AluminumDataset.AluminumDataset.anchorBox)
        fullLoss, confLoss, clsLoss, posLoss = loss
        Logger.log("fullLoss = {:.6f} (confLoss = {:.6f}, clsLoss = {:.6f}, posLoss = {:.6f})".format(fullLoss.item(), confLoss.item(), clsLoss.item(), posLoss.item()))
        epochTrainLoss += fullLoss.item() * img.size()[0]
        optimizer.zero_grad()
        fullLoss.backward()
        optimizer.step()
    epochTrainLossPerImage = epochTrainLoss / len(trainDataset)

    #Validate部分
    Logger.log("Validating...")
    model.eval()
    epochValidateLoss = 0.0
    with torch.no_grad():
        for i, data in enumerate(validateLoader, 1):
            Logger.log("[ Epoch:", epoch, "| Batch:", i, "]")
            img = data[0]
            label = data[1:]
            if useGPU:
                img = img.cuda()
                for j in range(0, 3):
                    label[j] = label[j].cuda()
            output = model(img)
            loss = criterion(output, label, YOLOv3_AluminumDataset.AluminumDataset.anchorBox)
            fullLoss, confLoss, clsLoss, posLoss = loss
            Logger.log("fullLoss = {:.6f} (confLoss = {:.6f}, clsLoss = {:.6f}, posLoss = {:.6f})".format(fullLoss.item(), confLoss.item(), clsLoss.item(), posLoss.item()))
            epochValidateLoss += fullLoss.item() * img.size()[0]
    epochValidateLossPerImage = epochValidateLoss / len(validateDataset)

    if useGPU:
        model.cpu()
    torch.save(model.state_dict(), "{}cnn_epoch{}.pth".format(outputPath, epoch))
    Logger.log("Finish epoch {}, TrainLoss: {:.6f}, ValidateLoss: {:.6f}".format(epoch, epochTrainLossPerImage, epochValidateLossPerImage))
