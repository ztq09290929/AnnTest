import numpy as np
import cv2
#encoding: utf - 8
#opencv方法
img = cv2.imread("C:\\Users\\loves_000\\Desktop\\制作相册、\\14.jpg")
cv2.imshow("bgr",img)
b,g,r = cv2.split(img)
dst = cv2.merge([b,g,r])
cv2.imshow('dst',dst)
cv2.imshow('b',b)
cv2.imshow('g',g)
cv2.imshow('r',r)
cv2.waitKey(0)
cv2.destroyAllWindows()
#numpy方法，做成了彩色
nb = np.zeros((img.shape[0],img.shape[1],3),dtype = img.dtype)
ng = np.zeros((img.shape[0],img.shape[1],3),dtype = img.dtype)
nr = np.zeros((img.shape[0],img.shape[1],3),dtype = img.dtype)

nb[:,:,0] = img[:,:,0]
ng[:,:,1] = img[:,:,1]
nr[:,:,2] = img[:,:,2]
    

cv2.imshow('nb',nb)
cv2.imshow('ng',ng)
cv2.imshow('nr',nr)
cv2.waitKey(0) 
cv2.destroyAllWindows()
