import time

"""
Logger的默认计时时间格式为”秒“
Logger默认不使用日志文件
"""

passedTimeFormat = 0 #0 sec, 1 hms
requireLogFile = False
logFilePath = ""
startTime = time.time()

# 这不是日志的一部分，所以无需考虑是否需要记录到日志文件里。
stStr = "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(startTime)) + "][0.00s] Logger Start Timing."
print(stStr)

def setPassedTimeFormat2sec():
    global passedTimeFormat
    passedTimeFormat = 0

def setPassedTimeFormat2hms():
    global passedTimeFormat
    passedTimeFormat = 1

def setLogFile(enable, filepath):
    global requireLogFile
    global logFilePath
    requireLogFile = enable
    logFilePath = filepath

def log(*args):
    global passedTimeFormat
    global requireLogFile
    global logFilePath
    global startTime
    currentTime = time.time()
    passedTime = currentTime - startTime
    strPassedTime = ""
    if passedTimeFormat == 0:
        strPassedTime = "{:.2f}s".format(passedTime)
    elif passedTimeFormat == 1:
        h = int(passedTime / 3600)
        m = int((passedTime - 3600 * h) / 60)
        s = passedTime - 3600 * h - 60 * m
        if passedTime < 60:
            strPassedTime = "{:.2f}s".format(s)
        elif passedTime >= 60 and passedTime < 3600:
            strPassedTime = "{}m {:.2f}s".format(m, s)
        elif passedTime >= 3600:
            strPassedTime = "{}h {}m {:.2f}s".format(h, m, s)

    tStr = "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(currentTime)) + "][" + strPassedTime + "]"
    print(tStr, end = " ")
    for i in range(0, len(args)):
        if i != len(args) - 1:
            print(args[i], end = " ")
        else:
            print(args[i])

    if requireLogFile:
        fp = open(file = logFilePath, mode = "a+", encoding = "UTF-8")
        print(tStr, end = " ", file = fp)
        for i in range(0, len(args)):
            if i != len(args) - 1:
                print(args[i], end = " ", file = fp)
            else:
                print(args[i], file = fp)
        fp.close()
