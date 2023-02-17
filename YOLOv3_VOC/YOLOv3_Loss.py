import torch
import torch.nn
import torch.optim
import torch.cuda

import math

class YOLOv3Loss(torch.nn.Module):
    def __init__(self, useGPU, lossScale = 1.0):
        super().__init__()
        self.useGPU = useGPU
        self.lossScale = lossScale

    def forward(self, output, label, anchorBox):
        t_output_A = output[0] # N, C, H, W
        t_output_A = t_output_A.permute(0, 2, 3, 1) # N, H, W, C
        s = t_output_A.size()
        t_output_A = t_output_A.reshape(s[0], s[1], s[2], 3, s[3] // 3) # N, R, C, B, D
        t_output_B = output[1] # N, C, H, W
        t_output_B = t_output_B.permute(0, 2, 3, 1) # N, H, W, C
        s = t_output_B.size()
        t_output_B = t_output_B.reshape(s[0], s[1], s[2], 3, s[3] // 3) # N, R, C, B, D
        t_output_C = output[2] # N, C, H, W
        t_output_C = t_output_C.permute(0, 2, 3, 1) # N, H, W, C
        s = t_output_C.size()
        t_output_C = t_output_C.reshape(s[0], s[1], s[2], 3, s[3] // 3) # N, R, C, B, D
        t_labelData_A = label[0] # 在416x416输入下，尺寸为32, 13, 13, 3, 25
        t_labelData_B = label[1]
        t_labelData_C = label[2]
        t_labelTag_A = label[3] # 在416x416输入下，尺寸为32, 300, 4
        t_labelTag_B = label[4]
        t_labelTag_C = label[5]

        Sigmoid = torch.nn.Sigmoid()
        BCELoss = torch.nn.BCELoss(reduction="none") #usage: BCELoss(input, target)
        MSELoss = torch.nn.MSELoss(reduction="none")


        batchPositionLoss = torch.tensor(0, dtype=torch.float32)
        batchClassLoss = torch.tensor(0, dtype=torch.float32)
        batchConfidenceLoss = torch.tensor(0, dtype=torch.float32)
        if self.useGPU:
            batchPositionLoss = batchPositionLoss.cuda()
            batchClassLoss = batchClassLoss.cuda()
            batchConfidenceLoss = batchConfidenceLoss.cuda()
        for m in range(0, 3):
            t_output_X = t_output_A
            t_labelData_X = t_labelData_A
            t_labelTag_X = t_labelTag_A
            if m == 0:
                t_output_X = t_output_A
                t_labelData_X = t_labelData_A
                t_labelTag_X = t_labelTag_A
            elif m == 1:
                t_output_X = t_output_B
                t_labelData_X = t_labelData_B
                t_labelTag_X = t_labelTag_B
            elif m == 2:
                t_output_X = t_output_C
                t_labelData_X = t_labelData_C
                t_labelTag_X = t_labelTag_C
            t_output_X[:, :, :, :, 0:2] = Sigmoid(t_output_X[:, :, :, :, 0:2])
            t_output_X[:, :, :, :, 4:] = Sigmoid(t_output_X[:, :, :, :, 4:])

            # 计算Confidence Loss
            t_output_X_clone = t_output_X.clone()  # not sure
            t_labelData_X_clone = t_labelData_X.clone()
            t_output_X_clone[:, :, :, :, 0:4] = torch.zeros(t_output_X_clone[:, :, :, :, 0:4].size(), dtype=torch.float32)
            t_labelData_X_clone[:, :, :, :, 0:4] = torch.zeros(t_labelData_X_clone[:, :, :, :, 0:4].size(), dtype=torch.float32)
            t_output_X_clone[:, :, :, :, 5:] = torch.zeros(t_output_X_clone[:, :, :, :, 5:].size(), dtype=torch.float32)
            t_labelData_X_clone[:, :, :, :, 5:] = torch.zeros(t_labelData_X_clone[:, :, :, :, 5:].size(), dtype=torch.float32)
            confidenceLoss = BCELoss(t_output_X_clone, t_labelData_X_clone)
            t_mask = torch.zeros(confidenceLoss.size(), dtype=torch.float32)
            t_mask[:, :, :, :, 4] = torch.ones(t_mask[:, :, :, :, 4].size(), dtype=torch.float32)
            for n in range(0, t_labelTag_X.size()[0]):
                for i in range(0, t_labelTag_X.size()[1]):
                    if t_labelTag_X[n][i][0].item() != -1:
                        if t_labelTag_X[n][i][3].item() == 1: #”忽略“
                            r = t_labelTag_X[n][i][0].item()
                            c = t_labelTag_X[n][i][1].item()
                            b = t_labelTag_X[n][i][2].item()
                            t_mask[n][r][c][b][4] = torch.tensor(0, dtype=torch.float32)
                    else:
                        break
            if self.useGPU:
                t_mask = t_mask.cuda()
            confidenceLoss = torch.sum(torch.mul(confidenceLoss, t_mask))

            # 计算Class Loss
            t_output_X_clone = t_output_X.clone()  # not sure
            t_labelData_X_clone = t_labelData_X.clone()
            t_output_X_clone[:, :, :, :, 0:5] = torch.zeros(t_output_X_clone[:, :, :, :, 0:5].size(), dtype=torch.float32)
            t_labelData_X_clone[:, :, :, :, 0:5] = torch.zeros(t_labelData_X_clone[:, :, :, :, 0:5].size(), dtype=torch.float32)
            classLoss = BCELoss(t_output_X_clone, t_labelData_X_clone)
            t_mask = torch.zeros(classLoss.size(), dtype=torch.float32)
            for n in range(0, t_labelTag_X.size()[0]):
                for i in range(0, t_labelTag_X.size()[1]):
                    if t_labelTag_X[n][i][0].item() != -1:
                        if t_labelTag_X[n][i][3].item() == 2: #”负责预测某个真实边界框“
                            r = t_labelTag_X[n][i][0].item()
                            c = t_labelTag_X[n][i][1].item()
                            b = t_labelTag_X[n][i][2].item()
                            t_mask[n][r][c][b][5:] = torch.ones(t_mask[n][r][c][b][5:].size(), dtype=torch.float32)
                    else:
                        break
            if self.useGPU:
                t_mask = t_mask.cuda()
            classLoss = torch.sum(torch.mul(classLoss, t_mask))

            # 计算Position Loss
            t_output_X_clone = t_output_X.clone()  # not sure
            t_labelData_X_clone = t_labelData_X.clone()
            t_output_X_clone[:, :, :, :, 4:] = torch.zeros(t_output_X_clone[:, :, :, :, 4:].size(), dtype=torch.float32)
            t_labelData_X_clone[:, :, :, :, 4:] = torch.zeros(t_labelData_X_clone[:, :, :, :, 4:].size(), dtype=torch.float32)
            positionLoss = MSELoss(t_output_X_clone, t_labelData_X_clone)
            t_mask = torch.zeros(positionLoss.size(), dtype=torch.float32)
            for n in range(0, t_labelTag_X.size()[0]):
                for i in range(0, t_labelTag_X.size()[1]):
                    if t_labelTag_X[n][i][0].item() != -1:
                        if t_labelTag_X[n][i][3].item() == 2: #”负责预测某个真实边界框“
                            r = t_labelTag_X[n][i][0].item()
                            c = t_labelTag_X[n][i][1].item()
                            b = t_labelTag_X[n][i][2].item()
                            lambda_pos = 2 - (anchorBox[m][b][0] * math.exp(t_labelData_X_clone[n][r][c][b][2].item())) * (anchorBox[m][b][1] * math.exp(t_labelData_X_clone[n][r][c][b][3].item()))
                            t_mask[n][r][c][b][0] = torch.tensor(lambda_pos, dtype=torch.float32)
                            t_mask[n][r][c][b][1] = torch.tensor(lambda_pos, dtype=torch.float32)
                            t_mask[n][r][c][b][2] = torch.tensor(lambda_pos, dtype=torch.float32)
                            t_mask[n][r][c][b][3] = torch.tensor(lambda_pos, dtype=torch.float32)
                    else:
                        break
            if self.useGPU:
                t_mask = t_mask.cuda()
            positionLoss = torch.sum(torch.mul(positionLoss, t_mask))

            batchConfidenceLoss = torch.add(batchConfidenceLoss, confidenceLoss)
            batchClassLoss = torch.add(batchClassLoss, classLoss)
            batchPositionLoss = torch.add(batchPositionLoss, positionLoss)

        t_batchSize = torch.tensor(t_labelTag_A.size()[0], dtype=torch.float32)
        if self.useGPU:
            t_batchSize = t_batchSize.cuda()
        avgConfidenceLoss = torch.div(batchConfidenceLoss, t_batchSize)
        avgClassLoss = torch.div(batchClassLoss, t_batchSize)
        avgPositionLoss = torch.div(batchPositionLoss, t_batchSize)

        #Loss缩放
        t_lossScale = torch.tensor(self.lossScale, dtype=torch.float32)
        if self.useGPU:
            t_lossScale = t_lossScale.cuda()
        avgConfidenceLoss = torch.mul(avgConfidenceLoss, t_lossScale)
        avgClassLoss = torch.mul(avgClassLoss, t_lossScale)
        avgPositionLoss = torch.mul(avgPositionLoss, t_lossScale)

        avgSingleImageLoss = torch.add(torch.add(avgConfidenceLoss, avgClassLoss), avgPositionLoss)
        return avgSingleImageLoss, avgConfidenceLoss, avgClassLoss, avgPositionLoss
