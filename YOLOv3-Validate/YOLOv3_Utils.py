import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import random

def drawBBox(img, format, bboxInfo, color):
    #img: pillow图像
    #format: 格式值，可以为0, 1
    #bboxInfo: 二维列表
    #   当format = 0，结构为[[xmin, ymin, xmax, ymax, classname],[...], ...]
    #   当format = 1，结构为[[xmin, ymin, xmax, ymax, classname, score], [...], ...]
    #color: 格式为(R, G, B)

    draw = PIL.ImageDraw.Draw(img)
    font_consola = PIL.ImageFont.truetype("./fonts/consola.ttf", 20)
    for obj in bboxInfo:
        draw.rectangle(xy = (obj[0], obj[1], obj[2], obj[3]), fill = None, outline = color, width = 2)
        objText = ""
        if format == 0:
            objText = obj[4]
        elif format == 1:
            objText = "{} [{:.2f}]".format(obj[4], obj[5])
        draw.text(xy = (obj[0], obj[1] - 20), text = objText, fill = color, font = font_consola)
    return img

def getRandomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
