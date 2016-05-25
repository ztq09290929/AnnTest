#coding=utf-8
import functions as KNN
import functionsCV as cv
import cv2
from numpy import *

#首先完成样本提取工作
print "Get samples..."
dataSet,labels = KNN.getSamples()#占用了大量时间
print "Get samples successfully!"
#获取摄像头
capture = cv2.VideoCapture(1)
success,frame = capture.read()
#进入循环，抓取图像
while success:
    success,frame = capture.read()
    #设置ROI区域
    cv2.rectangle(frame,(int(3*frame.shape[1]/8),int(3*frame.shape[0]/8)),(int(5*frame.shape[1]/8),int(5*frame.shape[0]/8)),[0,0,255])
    cv2.imshow('frame',frame)
    if (cv2.waitKey(1) == 27):
        capture.release()
        break
cv2.destroyAllWindows()

#截取ROI部分
img = frame[int(3*frame.shape[0]/8)+2:int(5*frame.shape[0]/8)-1,int(3*frame.shape[1]/8)+2:int(5*frame.shape[1]/8)-1,:].copy()

#提取包围数字的最小ROI,转化为32*32大小，并存入一个向量之中
roi = cv.findROI(img)
roi32 = cv.roiTo32(roi)
cv2.imshow('roi',roi32)
cv2.waitKey()
cv2.destroyAllWindows()
vec1024 = cv.roi2Vect(roi32)

#开始分类
print "Start finding..."
result =  KNN.classify0(vec1024,dataSet,labels,3)
print 'result: ',result
