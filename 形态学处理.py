#coding=utf-8
import cv2
import numpy as np

#定义结构元素
element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

img = cv2.imread(r'D:\python27\test\img.jpg')
#腐蚀
eroded = cv2.erode(img,element)
cv2.imshow('eroded',eroded)
#膨胀
dilated = cv2.dilate(img,element)
cv2.imshow('dilated',dilated)
#开运算
opened = cv2.morphologyEx(img,cv2.MORPH_OPEN,element)
cv2.imshow('opened',opened)
#闭运算
closed = cv2.morphologyEx(img,cv2.MORPH_CLOSE,element)
cv2.imshow('closed',closed)

#获得边缘
dilated = cv2.cvtColor(dilated,cv2.COLOR_BGR2GRAY)
eroded = cv2.cvtColor(eroded,cv2.COLOR_BGR2GRAY)
result = cv2.absdiff(dilated,eroded)#相减求边缘
ertval,result = cv2.threshold(result,40,255,cv2.THRESH_BINARY)#二值化处理
result = cv2.bitwise_not(result)#反色
cv2.imshow('result',result)


cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
