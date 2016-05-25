from numpy import *
import cv2
from os import listdir
#encoding: utf - 8
#获取文件行列数



def file2Image(inputPath,inputName,outputPath):
    fr = open(inputPath+'\\'+inputName,'r')
    col = len(fr.readline().strip('\n').strip('\r'))
    fr.close()
    fr = open(inputPath+'\\'+inputName,'r')
    lis = fr.readlines()
    row = len(lis)
    #创建一个等大的全黑图像
    img = zeros((row,col),dtype = uint8)
    #用文件给图像赋值
    for i in range(row):
        for j in range(col):
            img[i,j] = int(lis[i][j])*255
    outputName = inputName.split('.')[0]+'.jpg'
    cv2.imwrite(outputPath+'\\'+outputName,img)
#开始操作
inputPath = 'F:\\硕士课程学习\\机器学习实战及配套代码\\machinelearninginaction\\Ch02\\testDigits'
outputPath = 'F:\\硕士课程学习\\机器学习实战及配套代码\\machinelearninginaction\\Ch02\\testImages'
fileList = listdir(inputPath)

for i in range(len(fileList)):
    file2Image(inputPath,fileList[i],outputPath)
