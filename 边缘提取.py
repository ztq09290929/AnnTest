#coding=utf-8
import cv2
import numpy as np

img = cv2.imread('img.jpg',0)

x = cv2.Sobel(img,cv2.CV_16S,1,0)#按x方向求一阶导数
y = cv2.Sobel(img,cv2.CV_16S,0,1)#按y方向求一阶导数

absX = cv2.convertScaleAbs(x)#该函数目标变量深度为8U
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
rev,dst = cv2.threshold(dst,0,255,cv2.THRESH_OTSU)

cv2.imshow('absX',absX)
cv2.imshow('absY',absY)

cv2.imshow('dst',dst)

cv2.waitKey()
cv2.destroyAllWindows()

img = cv2.imread('img.jpg',0)
canny = cv2.Canny(img,50,150)
cv2.imshow('canny',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
