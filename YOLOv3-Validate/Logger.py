import time

#Logger的默认计时时间格式为“秒”
#Logger默认不使用日志文件

passedTimeFormat = 0 #0 sec, 1 hms
requireLogFile = False
logFilePath = ""
startTime = time.time()

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
    global startTime
    currentTime = time.time()
    passedTime = currentTime - startTime
    strPassedTime = ""
    if passedTimeFormat == 0:
        strPassedTime = "{0:.2f}s".format(passedTime)
    elif passedTimeFormat == 1:
        h = int(passedTime / 3600)
        m = int((passedTime - 3600 * h) / 60)
        s = passedTime - 3600 * h - 60 * m
        if h != 0:
            strPassedTime = strPassedTime + "{0}h ".format(h)
        if m != 0:
            strPassedTime = strPassedTime + "{0}m ".format(m)
        strPassedTime = strPassedTime + "{0:.2f}s".format(s)

    str = "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(currentTime)) + "][" + strPassedTime + "]"
    print(str, end = " ")
    for args_i in args:
        print(args_i, end = " ")
    print("")

    if requireLogFile:
        fp = open(file = logFilePath, mode = "a+", encoding = "UTF-8")
        print(str, end=" ", file = fp)
        for args_i in args:
            print(args_i, end=" ", file = fp)
        print("", file = fp)
        fp.close()

def initialLog():
    currentTime = time.time()
    str = "[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(currentTime)) + "][0.00s] Logger Started."
    print(str)

initialLog()