# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:15:29 2016

@author: ztq
"""
import numpy as np
import network
import mnist_loader
import saveData
import functionsCV as cv
import cv2
    
if __name__ == '__main__':
    bTrain = True
    if bTrain:    
        trainingdata, validationdata, testdata = mnist_loader.load_data_wrapper()
        #trainingdata, testdata = mnist_loader.load_data_wrapper_my()
        print 'The sum of trainingdata is: ',len(trainingdata)
        print 'The sum of testdata is: ',len(testdata)
        net = network.Network([784,30,10])
        #net = network.Network([1024,30,10])
        print 'The num of neural :',net.sizes
        print 'Start training...'
        #net.SGD(trainingdata, 40, 10, 1.2, test_data=testdata , weight_decay = 0)
        net.SGD(trainingdata, 30, 10, 0.5, test_data=testdata , weight_decay = 0.00005)#weight_decay权重衰减等同于加正则项解决过拟合
        #net.SGD(trainingdata, 30, 5, 1.5, test_data=testdata)
        print 'Training complete!'
        saveData.storeData([net.weights, net.biases],'AnnTrainedParas.txt')
        #saveData.storeData([net.weights, net.biases],'AnnTrainedParasMy.txt')
        print 'Saving paras complete! \nThe form is list[weights, biases].'       
        net.show_results()
    
    else:
        #net2 = network.Network([784,30,10])
        net2 = network.Network([1024,30,10])
        print 'before:',net2.biases
        #paras = saveData.grabData('AnnTrainedParas.txt')
        paras = saveData.grabData('AnnTrainedParasMy.txt')
        net2.weights = paras[0]
        net2.biases = paras[1]
        print 'after:',net2.biases
        print "各层偏置的维数："
        for a in net2.biases:
            print a.shape
        print "各层权值的维数："
        for b in net2.weights:
            print b.shape
            
        while(True):
            #获取摄像头
            capture = cv2.VideoCapture(0)
            success,frame = capture.read()
            print "进行一次抓拍，按ESC完成抓拍"
            #进入循环，抓取图像
            while success:
                success,frame = capture.read()
                #设置ROI区域
                cv2.rectangle(frame,(int(3*frame.shape[1]/8),int(3*frame.shape[0]/8)),(int(5*frame.shape[1]/8),int(5*frame.shape[0]/8)),[0,0,255])
                cv2.imshow('frame',frame) 
                if (cv2.waitKey(1) == 1048603):
                #if (cv2.waitKey(1) == 27):
                    capture.release()
                    break
            cv2.destroyAllWindows()
        
            #截取ROI部分
            img = frame[int(3*frame.shape[0]/8)+2:int(5*frame.shape[0]/8)-1,int(3*frame.shape[1]/8)+2:int(5*frame.shape[1]/8)-1,:].copy()
        
            roi = cv.findROI(img)
            #提取包围数字的最小ROI,转化为28*28大小，并存入一个向量之中
            #roi28 = cv.roiTo28(roi)
           # vec784 = cv.roi2Vect784(roi28)
            
            #提取包围数字的最小ROI,转化为32*32大小，并存入一个向量之中
            roi32 = cv.roiTo32(roi)
            vec1024 = cv.roi2Vect1024(roi32)
        
            #开始分类
            print "识别结果： ",net2.predict(vec1024)       
        
            #显示32*32图像，并等待循环
            cv2.imshow('roi',roi32)
            char = cv2.waitKey()
            cv2.destroyAllWindows()
            if char == 1048603 :break
            #if char == 27 :break
    
    
