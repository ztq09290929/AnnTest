#coding=utf-8
from numpy import *

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName = 'trainingDigits'):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def annClassifyTrain(dataArr,labelArr,num = 2000):
    dataMatrix = mat(dataArr)

    targetArray = array([[0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],\
                         [0.1,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],\
                         [0.1,0.1,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1],\
                         [0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1,0.1,0.1],\
                         [0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1,0.1],\
                         [0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1],\
                         [0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1],\
                         [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1],\
                         [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1],\
                         [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9]])
    print targetArray
    weightsH = zeros((3,1025)) #1024维输入 + w0的常量偏置
    weightsO = zeros((10,4))    #3维隐藏单元 + w0的常量偏置
    dweightsH = zeros((3,1025)) #记录本次w的修改量，供下次的冲量项使用
    dweightsO = zeros((10,4))

    m,n = shape(dataMatrix)
    for j in range(num):#共训练num次
        print 'j=',j
        dataIndex = range(m)
        for i in range(m):#对所有样本进行遍历
            yita = 4/(1.0+j+i)+0.01#随着迭代次数的增加，学习率逐渐降低，会缓解数据波动
            alpha = 0.3
            randIndex = int(random.uniform(0,len(dataIndex)))#内循环中每次随机选择一个样本进行训练
            weightsH,weightsO,dweightsH,dweightsO = updateWeights(\
                dataMatrix[dataIndex[randIndex],:],targetArray[labelArr[dataIndex[randIndex]],:],weightsH,weightsO,dweightsH,dweightsO,yita,alpha)
            del(dataIndex[randIndex])#排除此次使用的样本
    return weightsH,weightsO

def updateWeights(dataMat,targetArr,weightsH,weightsO,dweightsH,dweightsO,yita,alpha):

    outputH = zeros((1,3))
    outputK = zeros((1,10))
    errorsH = zeros((1,3))
    errorsK = zeros((1,10))
  
    for i in range(3):
        outputH[0,i] = sigmoid(mat(weightsH[i,1:])*dataMat.T+weightsH[i,0])
    for i in range(10):
        outputK[0,i] = sigmoid(sum(weightsO[i,1:]*outputH)+weightsO[i,0])
    errorsK = outputK*(1-outputK)*(targetArr - outputK)
    for j in range(3):
        errorsH[0,j] = outputH[0,j]*(1-outputH[0,j])*sum(weightsO[:,j+1]*errorsK)
    for i in range(3):
        dweightsH[i,1:] = alpha*dweightsH[i,1:]+yita*errorsH[0,i]*array(dataMat)
        dweightsH[i,0] = alpha*dweightsH[i,0]+yita*errorsH[0,i]*1
    for i in range(10):
        dweightsO[i,1:] = alpha*dweightsO[i,1:]+yita*errorsK[0,i]*outputH
        dweightsO[i,0] = alpha*dweightsO[i,0]+yita*errorsK[0,i]*1
    weightsH = weightsH + dweightsH
    weightsO = weightsO + dweightsO
    return weightsH,weightsO,dweightsH,dweightsO

def annClassify(weightsH,weightsO,dataArr):
    outputH = zeros((1,3))
    outputK = zeros((1,10))
  
    for i in range(3):
        outputH[0,i] = sigmoid(sum(weightsH[i,1:]*dataArr)+weightsH[i,0])
    for i in range(10):
        outputK[0,i] = sigmoid(sum(weightsO[i,1:]*outputH)+weightsO[i,0])
    result = argmax(outputK[0,:])
    return outputK[0,:]
    
    
