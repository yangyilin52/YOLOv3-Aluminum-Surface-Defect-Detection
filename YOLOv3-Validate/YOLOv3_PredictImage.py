import YOLOv3_CorePredict
import YOLOv3_Utils
import PIL.Image
import os
import cv2
import numpy

while True:
    imageFilePath = input("请输入要检测的图片的路径: ")
    if os.path.exists(imageFilePath):
        pillowImage = PIL.Image.open(imageFilePath)
        print("图片大小: {}x{}".format(pillowImage.width, pillowImage.height))
        result = YOLOv3_CorePredict.predict(pillowImage, 416)
        YOLOv3_Utils.drawBBox(pillowImage, 1, result, (0, 255, 0))
        cvImage = cv2.cvtColor(numpy.asarray(pillowImage), cv2.COLOR_RGB2BGR)
        windowsName = "Result"
        cv2.imshow(windowsName, cvImage)
        cv2.waitKey()
        cv2.destroyWindow(windowsName)
    else:
        print("文件不存在！")
    print("")
