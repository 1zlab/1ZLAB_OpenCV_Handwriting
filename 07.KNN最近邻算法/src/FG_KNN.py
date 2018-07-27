 
# 导入数据库
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import collections
import itertools


def drawData(data, target):
    '''
    绘制数据集
    '''
    targetColorDict = {
        0: 'red',
        1: 'green',
        2: 'blue'
    }

    for t,color in targetColorDict.items():
        idxs = np.argwhere(target == t)
        idxs.resize(len(idxs))
        plt.scatter(data[idxs][:,0],data[idxs][:,1],s=25,marker='o', c=targetColorDict[t])

    plt.show()

def Normalize(data):
    '''
    对数据进行归一化处理
    '''
    n_dimension = len(data[0])
    new_data = np.copy(data)
    
    for d_idx in range(n_dimension):
        minValue = np.min(data[:,d_idx])
        maxValue = np.max(data[:,d_idx])
        new_data[:,d_idx] = (data[:,d_idx] - minValue)/(maxValue - minValue)
    return new_data

def EuclideanDistance(data, vect):
    '''
    计算一个点与其他点集的距离，输出一个距离列表
    '''
    return np.sqrt(((data-vect)**2).sum(axis=1))


def KNN(data, vect, K=3):
    '''
    K 邻近算法的实现
    '''
    distance = EuclideanDistance(data, vect)
    kn_idxs = distance.argsort()[:K]
    # 获取前K个邻居的标签
    kn_labels = target[kn_idxs]
    # 对标签进行统计
    kn_labels_stat = dict(collections.Counter(kn_labels))
    # 获取数量最高的标签
    predict = max(kn_labels_stat, key=lambda x: kn_labels_stat[x])
    return predict


# 从数据库中载入iris样本
iris = datasets.load_iris()

X = iris.data
y = iris.target


# 打乱排序

data = X[:,:2] # 只提取前两维数据
n_sample = len(data)
idxs = np.arange(n_sample) # 初始化排序
np.random.shuffle(idxs) # 打乱排序

data = data[idxs]
target = y[idxs]



data = Normalize(data)
drawData(data, target)



targetColorDict = {
    0: 'red',
    1: 'green',
    2: 'blue'
}

K = 3

print('下方操作非常耗时，请耐心等待')
# 绘制预测点
for pt in itertools.product(np.arange(0,1,0.02), np.arange(0,1,0.02)):
    t = KNN(data, np.array([pt[0], pt[1]]), K)
    plt.scatter(np.array([pt[0]]),np.array([pt[1]]),s=25,marker='o', c=targetColorDict[t],alpha=0.2)

# 绘制数据点
for t,color in targetColorDict.items():
    idxs = np.argwhere(target == t)
    idxs.resize(len(idxs))
    plt.scatter(data[idxs][:,0],data[idxs][:,1],s=25,marker='o', c=targetColorDict[t])


plt.show()   
