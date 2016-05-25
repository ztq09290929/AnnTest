from numpy import *
import cv2
from os import listdir
#encoding: utf - 8
#��ȡ�ļ�������



def file2Image(inputPath,inputName,outputPath):
    fr = open(inputPath+'\\'+inputName,'r')
    col = len(fr.readline().strip('\n').strip('\r'))
    fr.close()
    fr = open(inputPath+'\\'+inputName,'r')
    lis = fr.readlines()
    row = len(lis)
    #����һ���ȴ��ȫ��ͼ��
    img = zeros((row,col),dtype = uint8)
    #���ļ���ͼ��ֵ
    for i in range(row):
        for j in range(col):
            img[i,j] = int(lis[i][j])*255
    outputName = inputName.split('.')[0]+'.jpg'
    cv2.imwrite(outputPath+'\\'+outputName,img)
#��ʼ����
inputPath = 'F:\\˶ʿ�γ�ѧϰ\\����ѧϰʵս�����״���\\machinelearninginaction\\Ch02\\testDigits'
outputPath = 'F:\\˶ʿ�γ�ѧϰ\\����ѧϰʵս�����״���\\machinelearninginaction\\Ch02\\testImages'
fileList = listdir(inputPath)

for i in range(len(fileList)):
    file2Image(inputPath,fileList[i],outputPath)
