import numpy as np
import csv, os
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd

# 数据预处理
listTrainData = []
for i in range(18):
    listTrainData.append([])

# 读取源文件
textTrain = open(os.path.join(os.path.dirname(__file__), "01-Data/train.csv"), "r", encoding="big5")
rowTrain = csv.reader(textTrain)
n_row = 0

# 转换为结构数据，按污染物种类分类
for r in rowTrain:
    if n_row != 0:
        for i in range(3,27):
            if r[i] != "NR":
                listTrainData[(n_row-1)%18].append(float(r[i]))
            else:
                listTrainData[(n_row-1)%18].append(float(0))
    n_row += 1
textTrain.close()

# 分割数据和标签
# 训练数据每小时记录一次，每月记录20天，则每月共有480个记录
# 根据测试要求，每10个小时为一组数据，尽可能的利用数据，可分为471组
listTrainX = []
listTrainY = []

for m in range(12):
    for i in range(471):
        listTrainX.append([])
        listTrainY.append(listTrainData[9][480*m+i+9])
        for p in range(18):
            for t in range(9):
                listTrainX[471*m+i].append(listTrainData[p][480*m+i+t])

# 读取测试数据
listTestData = []
textTest = open(os.path.join(os.path.dirname(__file__),"01-Data/test.csv"), "r", encoding="big5")
rowTest = csv.reader(textTest)
n_row = 0

for r in rowTest:
    if n_row % 18 == 0:
        listTestData.append([])
    for i in range(2,11):
        if r[i] == "NR":
            listTestData[n_row // 18].append(float(0))
        else:
            listTestData[n_row // 18].append(float(r[i]))
    n_row += 1
textTest.close()

arrayTestX = np.array(listTestData)
arrayTrainX = np.array(listTrainX)
arrayTrainY = np.array(listTrainY)

def GD(X, Y, W, eta, Iteration, lambdaL2):
    # 均方误差的梯度下降
    listCost=[]
    for iter in range(Iteration):
        arrayYhat = X.dot(W)
        arrayLoss = arrayYhat - Y
        arrayGradient = X.T.dot(arrayLoss)/X.shape[0]+lambdaL2*W
        W -= eta*arrayGradient
        arrayCost = np.sum(arrayLoss**2)/X.shape[0]
        listCost.append(arrayCost)
    return W, listCost

# 训练
arrayTrainX = np.concatenate((np.ones((arrayTrainX.shape[0],1)), arrayTrainX), axis=1)

lr = 1e-6
arrayW = np.zeros(arrayTrainX.shape[1])
arrayW_gd, listCost_gd = GD(X=arrayTrainX, Y=arrayTrainY, W=arrayW, eta=lr, Iteration=20000, lambdaL2=0)
# arrayW = np.zeros(arrayTrainX.shape[1])
# arrayW_gd_1, listCost_gd_1 = GD(X=arrayTrainX, Y=arrayTrainY, W=arrayW, eta=lr, Iteration=20000, lambdaL2=100)

arrayTestX = np.concatenate((np.ones((arrayTestX.shape[0],1)),arrayTestX), axis=1)
arrayPredictY_gd = np.dot(arrayTestX, arrayW_gd)

plt.plot(np.arange(len(arrayPredictY_gd)), arrayPredictY_gd, "g--")
plt.title("GD")
plt.xlabel("Test Data Index")
plt.ylabel("Predict Result")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "02-Output/Compare"))
plt.show()

