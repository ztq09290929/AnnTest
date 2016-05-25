#coding=utf-8
import cv2
import numpy as np

def calcAndDrawHist(image,color):
    hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros((256,256,3),dtype = np.uint8)
    hpt = int(0.9*256)

    for h in range(256):
        intensity = int(hist[h]/maxVal*hpt)
        cv2.line(histImg,(h,256),(h,256-intensity),color)
    return histImg

if __name__ == '__main__':
    img = cv2.imread(r'D:\python27\test\img.jpg')
    b,g,r = cv2.split(img)

    histImgB = calcAndDrawHist(b,[255,0,0])
    histImgG = calcAndDrawHist(g,[0,255,0])
    histImgR = calcAndDrawHist(r,[0,0,255])

    cv2.imshow('histImgB',histImgB)
    cv2.imshow('histImgG',histImgG)
    cv2.imshow('histImgR',histImgR)
    cv2.imshow('img',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

    
