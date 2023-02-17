import os, random, shutil

#可调参数
train_num = 2705
test_num = 300
val_num = 300
#--------------------------------


def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

allFileList = os.listdir("./All/")
fileNameList = []
for elm in allFileList:
    if elm[-4:] == ".jpg":
        fileNameList.append(elm[0:-4])
all_num = len(fileNameList)

if all_num >= train_num + test_num and test_num >= val_num:
    mkdir("./Train/")
    mkdir("./Test/")
    mkdir("./Validate/")
    for i in range(0, train_num):
        idx = random.randint(0, len(fileNameList) - 1)
        shutil.copyfile("./All/{}.jpg".format(fileNameList[idx]), "./Train/{}.jpg".format(fileNameList[idx]))
        shutil.copyfile("./All/{}.json".format(fileNameList[idx]), "./Train/{}.json".format(fileNameList[idx]))
        del fileNameList[idx]
    for i in range(0, test_num):
        idx = random.randint(0, len(fileNameList) - 1)
        shutil.copyfile("./All/{}.jpg".format(fileNameList[idx]), "./Test/{}.jpg".format(fileNameList[idx]))
        shutil.copyfile("./All/{}.json".format(fileNameList[idx]), "./Test/{}.json".format(fileNameList[idx]))
        del fileNameList[idx]

    testFileList = os.listdir("./Test/")
    fileNameList = []
    for elm in testFileList:
        if elm[-4:] == ".jpg":
            fileNameList.append(elm[0:-4])
    for i in range(0, val_num):
        idx = random.randint(0, len(fileNameList) - 1)
        shutil.copyfile("./Test/{}.jpg".format(fileNameList[idx]), "./Validate/{}.jpg".format(fileNameList[idx]))
        shutil.copyfile("./Test/{}.json".format(fileNameList[idx]), "./Validate/{}.json".format(fileNameList[idx]))
        del fileNameList[idx]
    print("Done")
else:
    print("Error Settings!")
