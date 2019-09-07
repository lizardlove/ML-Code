import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def crossVal(data, label, num):
    data=data.reshape(data.shape[0],-1)
    label=label.reshape(label.shape[0],-1)
    data = np.array_split(data, num)
    label = np.array_split(label, num)
    avgScore=np.zeros(num)
    for i in range(num):

        xData=np.vstack(data[0:i]+data[i+1:])
        xLabel=np.vstack(label[0:i]+label[i+1:])
        xLabel = xLabel[:,0]
        yData=data[i]
        yLabel=label[i]

        classifier=KNearestNeighbor()
        classifier.train(xData, xLabel)
        dist=classifier.compute_distances_no_loops(yData)
        res=computeAccuracy(classifier,dist, yLabel[:,0])
        for j in sorted(list(res.values())):
            avgScore[i]=avgScore[i]+j
            print('%d, ac=%f' % (i, j))
        avgScore[i]=avgScore[i]/len(list(res.values()))
        plt.subplot(num/2,2,i+1)
        plt.plot(list(res.keys()), list(res.values()), 'ob')

    print(avgScore.mean())
    plt.show()
    return 




# 计算正确率
def computeAccuracy(classifier, dist, yLabel):
    result={}
    kValue=range(1,20)
    for i in kValue:
        y_test_pred = classifier.predict_labels(dist, k=i)
        num_correct = np.sum(y_test_pred==yLabel)
        accuracy = float(num_correct) / len(yLabel)
        result[i]=accuracy
    return result


# 加载CIFAR-10数据集，按5-1划分
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
trainData, trainLabel, testData, testLabel = load_CIFAR10(cifar10_dir)

#取部分数据做训练数据
num_training=5000
mask=range(num_training)
trainData=trainData[mask]
trainLabel=trainLabel[mask]
num_test=500
mask=range(num_test)
testData=testData[mask]
testLabel=testLabel[mask]

crossVal(trainData, trainLabel,10)



