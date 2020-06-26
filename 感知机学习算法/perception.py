# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 21:21
# @Author  : EugeneLi
# @Email   : 861269630@qq.com
# @File    : perception.py


'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
每一张图片是28x28=784
然后将像素展平 存放在一个行向量里面
'''

import csv
import numpy as np
import time


def loadData(fileName):
    '''
    加载数据集Mnist 格式为.csv
    :param fileName: 数据集路径
    :return: 数据集的list以及标记
    '''
    print("start to read data")
    dataArr=[]#存放数据
    labelArr=[]#存放标签
    #打开文件
    fr=open(fileName,'r')
    #将文件按行读取
    for line in fr.readlines():
        #对每一行数据按照‘，’进行切割，返回字段列表
        curLine=line.strip().split(',')
        # 内容中下标为0的第一个数据存放的是种类0-9，
        # 该数据中有0-9个标记 因为是二分类的任务，所以>=5作为1 <5作为-1
        #print(curLine)
        if  int(curLine[0]) >= 5:
            labelArr.append(1)#存放正类1 即加入数组的数据是1
        else:
            labelArr.append(-1)
        # 对数据做归一化处理
        #[int(num) for num in  curLine[1:]] 遍历每一行中除了第一个元素之外的其他所有元素 转成int
        dataArr.append([int(num)/255 for num in curLine[1:]])
    #返回数据、标签
    return dataArr,labelArr

def perception(dataArr,labelArr,iter=50):
    '''
    感知器的训练过程
    :param dataArr:训练集的数据list
    :param labelArr:训练集标签
    :param iter:迭代次数 50
    :return: 训练好的w 和 b
    '''
    print('start to train')
    # 将数据转换成矩阵形式（在机器学习中因为通常都是向量的运算，转换称矩阵形式方便运算）
    # 转换后的数据中每一个样本的向量都是横向的
    dataMat=np.mat(dataArr)
    print('dataMat此时的shape',dataMat.shape)# dataMat此时的shape(60000,784)
    #将标签转化为矩阵 之后矩阵(.T为转置)
    #在运算中需要单独取label中的某一个元素 如果是1xN的矩阵的话 无法使用label[i]的方式读取
    labelMat=np.mat(labelArr).T#使得labelMath就是一个列向量一样 即(60000,1)
    print("labelMat此时的shape",labelMat.shape) #labelMat此时的shape (60000,1)
    #获取数据矩阵的大小 为m*n
    m,n=np.shape(dataMat)
    print("m n 为",(m,n)) #60000 784

    #开始进行感知机学习算法
    #创建初始权重w  初始值为0 n为np.shape(dataMat)[1]
    #控制样本长度保持一致
    w=np.zeros((1,np.shape(dataMat)[1]))
    print('w的shape',w.shape) #w的shape w的shape (1,784)
    #初始化偏置b为0
    b=0
    #初始化步长，也就是梯度下降的过程中的n 控制梯度下降速率
    h=0.0001
    #进行iter次迭代运算
    for k in range(iter):
        #对于每一个样本进行梯度下降
        for i in range(m):
            #获取当前样本的向量
            xi=dataMat[i]# (1, 784) print("此时xi是",xi.shape) #此时xi是 (1, 784)
            #获取当前样本所对应的标签
            yi=labelMat[i]#(1, 1) print("此时的yi是",yi.shape)#此时的yi是 (1, 1)
            #判断是否为误分类样本：yi(w*xi+b)<=0
            if yi*(w*xi.T+b)<=0:
                #对于误分类样本 进行梯度下降 更新w和b
                w=w+h*yi*xi
                b=b+h*yi
        #打印训练的进度
        print('当前为 %d/%d 训练' % (k, iter))
    #返回训练的w b
    return w,b


def test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr: 测试集
    :param labelArr: 测试集标签
    :param w:训练获得的权重
    :param b:训练获得的偏置
    :return:正确率
    '''
    print('start to test')
    #将数据集转换为矩阵形式方便运算
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).T

    #获取测试数据集矩阵的大小
    m,n=np.shape(dataMat)
    print("测试集的m n 为",(m,n))
    #错误样本数的计算
    errorCnt=0
    #遍历所有的测试样本
    for i in range(m):
        #获得单个样本向量
        xi=dataMat[i]#print('测试集中的xi',xi.shape)#(1, 784)
        yi=labelMat[i] #print('测试集中的yi',yi.shape)#(1, 1)
        #获得运算结果
        result=yi * (w * xi.T + b)
        if result<=0:errorCnt+=1 #result<=0 说明是误分类，所以错误的样本数+1
    #正确率=1-(样本分类错误数/样本总数)
    accuRate=1-(errorCnt/m)
        #返回正确率
    return accuRate

if __name__ == '__main__':
    #获取当前的时间 保存前后的时间 来记录算法运行的时间
    start=time.time()
    fileTrain='../Mnist/mnist_train.csv'
    fileTest='../Mnist/mnist_test.csv'
    #获取训练集以及标签
    trainData,trainLabel=loadData(fileTrain)
    #获取测试集以及标签
    testData,testLabel=loadData(fileTest)
    #训练得到w b
    w,b=perception(trainData,trainLabel,40)
    # 进行测试，获得正确率
    accuRate = test(testData, testLabel, w, b)
    print('正确率是',accuRate)
    end=time.time()
    print("the perception algorithm time is",(end-start))

