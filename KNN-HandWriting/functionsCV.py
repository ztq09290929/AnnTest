#coding=utf-8
import cv2
from numpy import *

#提取包含数字的ROI正方形
def findROI(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转化为灰度图
    retval, binary = cv2.threshold(gray, 130 , 255, cv2.THRESH_BINARY_INV);#二值化，前景为白色
    #element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    #dilate = cv2.dilate(binary,element)#用椭圆结构元素进行前景的膨胀
    binaryImg = binary.copy()#备份膨胀后的二值图像
    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#寻找轮廓
    if len(contours)>1:
        print '错误！轮廓数量多于一个！'
        return

    #查找轮廓上下左右的极值点
    leftMostX = int(contours[0][:,0,0].min())
    rightMostX = int(contours[0][:,0,0].max())
    topMostY = int(contours[0][:,0,1].min())
    downMostY = int(contours[0][:,0,1].max())
    #提取包围数字的最小ROI，且要求长宽相等
    roiBinary = binaryImg[topMostY-1:downMostY+2,leftMostX-1:rightMostX+2].copy()#切出数字部分
    oldWidth = rightMostX-leftMostX+3#原始数字宽
    width = downMostY - topMostY +3 #正方形化后的边长
    rectBinary = zeros((width,width),dtype = uint8)#创建正方形化之后的图像
    rectBinary[:,(width-oldWidth)/2:((width-oldWidth)/2+oldWidth)] = roiBinary.copy()#得到正方形图像
    
    return rectBinary
    
#将包含数字的ROI缩小为32*32
def roiTo32(roi):
    returnROI = zeros((32,32),dtype = uint8)
    cv2.resize(roi,(32,32),returnROI)
    retval, binary = cv2.threshold(returnROI, 1 , 255, cv2.THRESH_BINARY);
    return binary
    
#将32*32的图像转化为一个1024长度的只包含0和1的一维向量
def roi2Vect(roi):
    returnVect = zeros((1,1024))
    for i in range(32):
        for j in range(32):
            returnVect[0,i*32+j] = int(roi[i,j]/255)
    return returnVect
    
    
