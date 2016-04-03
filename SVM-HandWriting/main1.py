#coding=utf-8
import svmMLiA as svm
import functionsCV as cv
import cv2
from saveData import *
from numpy import *

#首先完成样本提取工作
print "Get samples..."
dataArr,labelArr = svm.loadImages('trainingDigits')
print "Get samples successfully!"
#开始训练
print "Start training..."
b,alphas = svm.smoP(dataArr,labelArr,200,0.0001,10000,('rbf',10))
storeData(b,'b.txt')
storeData(alphas,'alphas.txt')
print "Training completed!"

#寻找支持向量
dataMat = mat(dataArr)
labelMat = mat(labelArr).transpose()
svInd = nonzero(alphas.A>0)[0]#支持向量索引
sVs = dataMat[svInd]#支持向量
labelSV = labelMat[svInd]#支持向量的标签
storeData(svInd,'svInd.txt')
storeData(sVs,'sVs.txt')
storeData(labelSV,'labelSV.txt')
