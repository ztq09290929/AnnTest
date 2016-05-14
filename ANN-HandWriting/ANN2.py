# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:15:29 2016

@author: ztq
"""
import numpy as np
import network2
import mnist_loader
import functionsCV as cv
import cv2
    
if __name__ == '__main__':
    bTrain = True
    if bTrain:    
        trainingdata, validationdata, testdata = mnist_loader.load_data_wrapper()
        print 'The sum of trainingdata is: ',len(trainingdata)
        print 'The sum of validationdata is: ',len(validationdata)
        net = network2.Network([784,30,10],cost = network2.CrossEntropyCost)
        print 'The num of neural :',net.sizes
        print 'Start training...'
        ec,ea,tc,ta = net.SGD(trainingdata, 30, 10, 0.5, \
                        5.0,validationdata ,\
                        True,True,True,True)
        print 'Training complete!'
        net.save('AnnTrainedParas2.txt')
        print 'Saving paras complete! \nThe form is dict.'   
        ea = [item/float(len(validationdata)) for item in ea]
        ta = [item/float(len(trainingdata)) for item in ta]
        network2.show_result(ec,ea,tc,ta)
    
    else:
        net2 = network2.load('AnnTrainedParas2.txt')
        print 'after:',net2.biases
        print "各层偏置的维数："
        for a in net2.biases:
            print a.shape
        print "各层权值的维数："
        for b in net2.weights:
            print b.shape    
     
    
    
