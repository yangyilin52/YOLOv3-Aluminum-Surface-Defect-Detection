import cv2
import random
import numpy as np
import PIL.Image
import copy

printLog = False

#---------------------------------------------
#                    设置
#---------------------------------------------
def setEnablePrintLog(enable):
    global printLog
    printLog = enable


#---------------------------------------------
#                   必要操作
#---------------------------------------------
class StartAugmentation:
    def __init__(self):
        pass

    def __call__(self, PILImage, label):
        """
        参数:
            PILImage: PIL图片对象
            label: 边界框信息列表，格式为[[xmin(int), ymin(int), xmax(int), ymax(int), classname(str)], ......]
        功能:
            返回一个opencv图片对象和一个边界框信息列表。
            因为是深拷贝，所以对返回的图片和边界框信息列表进行操作，不会影响到输入的图片和边界框信息列表。
        """
        newImage = cv2.cvtColor(np.array(PILImage), cv2.COLOR_RGB2BGR)
        newLabel = copy.deepcopy(label)
        return newImage, newLabel

class EndAugmentation:
    def __init__(self):
        pass

    def __call__(self, cvImage, label):
        """
        参数:
            cvImage: opencv图片对象
            label: 边界框信息列表，格式为[[xmin(int), ymin(int), xmax(int), ymax(int), classname(str)], ......]
        功能:
            返回一个PIL图片对象和一个边界框信息列表。
        """
        image = PIL.Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
        return image, label


#---------------------------------------------
#    像素内容变换（Photometric Distortions）
#---------------------------------------------
class RandomBrightness:
    def __init__(self, betaRange = 32):
        self.isParamValid = False
        if betaRange >= 0:
            self.betaRange = betaRange
            self.isParamValid = True

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if self.isParamValid:
            if random.randint(0, 1) == 1:
                image = image.astype(np.float32)
                beta = random.uniform(-self.betaRange, self.betaRange)
                image += beta
                image = np.clip(image, 0, 255)
                image = image.astype(np.uint8)
                if printLog:
                    print("Random Brightness: beta={:.2f}".format(beta))
        return image, label

class RandomContrast:
    def __init__(self, minAlpha = 0.5, maxAlpha = 1.5):
        self.isParamValid = False
        if maxAlpha >= minAlpha and minAlpha > 0:
            self.minAlpha = minAlpha
            self.maxAlpha = maxAlpha
            self.isParamValid = True

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if self.isParamValid:
            if random.randint(0, 1) == 1:
                image = image.astype(np.float32)
                alpha = random.uniform(self.minAlpha, self.maxAlpha)
                image *= alpha
                image = np.clip(image, 0, 255)
                image = image.astype(np.uint8)
                if printLog:
                    print("Random Contrast: alpha={:.2f}".format(alpha))
        return image, label

class RandomSaturation:
    def __init__(self, minAlpha = 0.5, maxAlpha = 1.5):
        self.isParamValid = False
        if maxAlpha >= minAlpha and minAlpha > 0:
            self.minAlpha = minAlpha
            self.maxAlpha = maxAlpha
            self.isParamValid = True

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if self.isParamValid:
            if random.randint(0, 1) == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image = image.astype(np.float32)
                alpha = random.uniform(self.minAlpha, self.maxAlpha)
                image[:, :, 1] *= alpha
                image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                if printLog:
                    print("Random Saturation: alpha={:.2f}".format(alpha))
        return image, label

class RandomHue:
    def __init__(self, betaRange = 9):
        self.isParamValid = False
        if betaRange >= 0 and betaRange <= 180:
            self.betaRange = betaRange
            self.isParamValid = True

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if self.isParamValid:
            if random.randint(0, 1) == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                image = image.astype(np.float32)
                beta = random.uniform(-self.betaRange, self.betaRange)
                image[:, :, 0] += beta
                image[:, :, 0][image[:, :, 0] > 180.0] -= 180.0
                image[:, :, 0][image[:, :, 0] < 0.0] += 180.0
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
                if printLog:
                    print("Random Hue: beta={:.2f}".format(beta))
        return image, label

class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if random.randint(0, 1) == 1:
            swap = self.perms[random.randint(0, 5)]
            image[:, :, (0, 1, 2)] = image[:, :, swap]
            if printLog:
                print("Random Lighting Noise: swap={}".format(swap))
        return image, label


#---------------------------------------------
#     空间几何变换（Geometric Distortions）
#---------------------------------------------
class RandomMirror:
    def __init__(self):
        pass

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if random.randint(0, 1) == 1:
            choice = random.randint(0, 2)
            if choice == 0:
                #水平翻转
                image = cv2.flip(image, 1)
                newLabel = []
                width = image.shape[1]
                height = image.shape[0]
                for elm in label:
                    tl = []
                    tl.append((width - 1) - elm[2])
                    tl.append(elm[1])
                    tl.append((width - 1) - elm[0])
                    tl.append(elm[3])
                    tl.append(elm[4])
                    newLabel.append(tl)
                label = newLabel
                if printLog:
                    print("Random Mirror: horizontal")
            elif choice == 1:
                # 垂直翻转
                image = cv2.flip(image, 0)
                newLabel = []
                width = image.shape[1]
                height = image.shape[0]
                for elm in label:
                    tl = []
                    tl.append(elm[0])
                    tl.append((height - 1) - elm[3])
                    tl.append(elm[2])
                    tl.append((height - 1) - elm[1])
                    tl.append(elm[4])
                    newLabel.append(tl)
                label = newLabel
                if printLog:
                    print("Random Mirror: vertical")
            elif choice == 2:
                # 水平垂直翻转
                image = cv2.flip(image, -1)
                newLabel = []
                width = image.shape[1]
                height = image.shape[0]
                for elm in label:
                    tl = []
                    tl.append((width - 1) - elm[2])
                    tl.append((height - 1) - elm[3])
                    tl.append((width - 1) - elm[0])
                    tl.append((height - 1) - elm[1])
                    tl.append(elm[4])
                    newLabel.append(tl)
                label = newLabel
                if printLog:
                    print("Random Mirror: horizontal & vertical")
        return image, label

class RandomResize:
    def __init__(self, minWidthScale = 0.5, maxWidthScale = 1.0, minHeightScale = 0.5, maxHeightScale = 1.0, minArea = 25):
        self.isParamValid = False
        if maxWidthScale >= minWidthScale and minWidthScale > 0 and maxHeightScale >= minHeightScale and minHeightScale > 0:
            self.minWidthScale = minWidthScale
            self.maxWidthScale = maxWidthScale
            self.minHeightScale = minHeightScale
            self.maxHeightScale = maxHeightScale
            self.minArea = minArea
            self.isParamValid = True

    def __call__(self, cvImage, label):
        global printLog
        image = cvImage
        if self.isParamValid:
            if random.randint(0, 1) == 1:
                width = image.shape[1]
                height = image.shape[0]
                widthScale = random.uniform(self.minWidthScale, self.maxWidthScale)
                heightScale = random.uniform(self.minHeightScale, self.maxHeightScale)
                newSize = (int(width * widthScale), int(height * heightScale))
                image = cv2.resize(image, newSize, interpolation = cv2.INTER_LINEAR)
                newLabel = []
                for elm in label:
                    tl = []
                    tl.append(int(elm[0] * widthScale))
                    tl.append(int(elm[1] * heightScale))
                    tl.append(int(elm[2] * widthScale))
                    tl.append(int(elm[3] * heightScale))
                    tl.append(elm[4])
                    area = (tl[2] - tl[0]) * (tl[3] - tl[1])
                    if area > self.minArea:
                        newLabel.append(tl)
                label = newLabel
                if printLog:
                    print("Random Resize: (w={:.2f}, h={:.2f})".format(widthScale, heightScale))
        return image, label
