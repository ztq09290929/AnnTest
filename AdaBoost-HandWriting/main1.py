#coding=utf-8
import adaboost 
import functionsCV as cv
import cv2
from saveData import *
from numpy import *

#首先完成样本提取工作
print "Get samples..."
dataArr,labelArr = adaboost.loadImages('trainingDigits')
print "Get samples successfully!"
#开始训练
print "Start training..."
weakClassArr,aggClassEst = adaboost.adaBoostTrainDS(dataArr,labelArr,1000)
print "len = ",len(weakClassArr)
print weakClassArr
storeData(weakClassArr,'weakClassArr.txt')
print "Training completed!"


