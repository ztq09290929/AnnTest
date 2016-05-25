#coding=utf-8
import ann 
import functionsCV as cv
import cv2
from saveData import *
from numpy import *

#首先完成样本提取工作
print "Get samples..."
dataArr,labelArr = ann.loadImages('trainingDigits')
print "Get samples successfully!"
#开始训练
print "Start training..."
wh,wo = ann.adaBoostTrainDS(dataArr,labelArr,3000)
print "saving datas..."
storeData(wh,'wh.txt')
storeData(wo,'wo.txt')
print "Training completed!"


