# -*- coding: utf-8 -*-
# @Time    : 2020/7/2 15:48
# @Author  : EugeneLi
# @Email   : 861269630@qq.com
# @File    : knn.py

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000（实际使用：200）
运行结果：（邻近k数量：25）
正确率是 0.97 使用欧式距离的情况下
运行时间 475.7980799674988s
'''
import numpy as np
import time


def loadData(fileName):
    '''
    加载数据
    :param fileName: 数据集地址
    :return: 处理好的数据列表
    '''

    print('开始加载数据')
    dataArr=[]
    labelArr=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split(',')
        dataArr.append([int (num)for num in curLine[1:]])
        labelArr.append(int (curLine[0]))
    return dataArr,labelArr

def calcDist(x1,x2):
    '''
    计算两个样本向量之间的距离
    使用的是欧式距离 即样本中每个元素相减的平方 再求和 再开根号
    :param x1: 向量x1
    :param x2: 向量x2
    :return: 向量之间的距离
    '''
    # 马哈顿距离计算公式
    # return np.sum(x1 - x2)
    return np.sqrt(np.sum(np.square(x1-x2)))

def getClosest(trainDataMat,trainLabelMat,x,topK):
    '''
    预测样本x的标记
    获取方式通过查找与样本最近的topK个点，并查看他们的标签
    查找里面占据某类标签最多的那一类标签
    :param trainDataMat:训练集数据集
    :param trainLabelMat:训练集标签集
    :param x:要预测的样本x
    :param topK:选择参考最近邻的样本数目
    :return:预测的标记
    '''
    #建立一个存放向量x与每个训练集中样本距离的列表
    #列表的长度为训练集的长度 distList[i] 表示x与训练集中第i个样本的距离
    distList=[0]*len(trainLabelArr)#初始化
    for i in range(len(trainDataMat)):
        #获取训练集中当前样本的向量
        x1=trainDataMat[i]
        #计算向量x与训练集样本x的距离
        #print('x1 shape',x1.shape)#（1,784）
        #print('x shape',x.shape)#（1,784）
        curDist=calcDist(x1,x)
        #将距离放入对应的列表位置中
        distList[i]=curDist
    # 对距离列表进行排序
    # argsort：函数将数组的值从小到大排序后，并按照其相对应的索引值输出
    # 例如：
    #   >>> x = np.array([3, 1, 2])
    #   >>> np.argsort(x)
    #   array([1, 2, 0])
    # 返回的是列表中从小到大的元素索引值，对于我们这种需要查找最小距离的情况来说很合适
    # array返回的是整个索引值列表，我们通过[:topK]取列表中前topL个放入list中。
    topKList=np.argsort(np.array(distList))[:topK] #升序排序 取前k个
    # 建立一个长度10的列表 因为0-9是10类，用于选择数量最多的标记
    # 分类决策使用的是投票表决，topK个标记每人有一票，在数组中每个标记代表的位置中投入
    # 自己对应的地方，随后进行唱票选择最高票的标记
    labelList=[0]*10
    for index in topKList:
        #trainLabelMat[index] 在训练集标签中寻找topK元素索引对应的标记 然后找到标记在lebalList中对应的位置
        pos=int(trainLabelMat[index])
        labelList[pos]+=1
    #找到选票箱中票数最多的票数值
    #再返回对应最大值的索引 即等同于预测的标记
    return labelList.index(max(labelList))

def test(trainDataArr,trainLabelArr,testDataArr,testLabelArr,topK):
    '''
    测试正确率
    :param trainDataArr:
    :param trainLabelArr:
    :param testDataArr:
    :param testLabelArr:
    :param topK:选择多少个邻近点参考
    :return:
    '''

    print('开始测试')
    #将向量转化为矩阵的形式 标签记得转置
    trainDataMat=np.mat(trainDataArr)
    trainLabelMat=np.mat(trainLabelArr).T
    testDataMat=np.mat(testDataArr)
    testLabelMat=np.mat(testLabelArr).T
    print('trainDataMat shape',trainDataMat.shape)
    print('trainLabelMath shape', trainLabelMat.shape)
    print('testDataMat shape', testDataMat.shape)
    print('testDataMat shape', testDataMat.shape)
    #错误值的个数
    error=0
    #遍历数据集 对每个数据集进行测试
    #由于计算向量与向量之间的耗费太大了  选择使用200个测试集就好
    #for i in range(len(testDataMat)):
    for i in range(200):
        print('test %d:%d'%(i,200))
        #获取当前测试集的样本的向量
        x=testDataMat[i]
        #获取预测的标记
        y=getClosest(trainDataMat,trainLabelMat,x,topK)
        #如果预测标记与实际不符，错误值+1
        if y!=testLabelMat[i]:error+=1
    #返回正确率
    return 1-(error/200)


if __name__ == '__main__':
    begin=time.time()
    trainFileName="D:\AI\Learn\Mnist\mnist_train.csv"
    testFileName="D:\AI\Learn\Mnist\mnist_test.csv"
    #获取训练集
    trainDataArr,trainLabelArr=loadData(trainFileName)
    #获取测试集
    testDataArr,testLabelArr=loadData(testFileName)
    #计算正确率
    accur=test(trainDataArr,trainLabelArr,testDataArr,testLabelArr,25)
    print("正确率是",accur)
    end=time.time()
    print('运行时间',end-begin)