# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:15:29 2016

@author: ztq
"""
import numpy as np
import network
import mnist_loader
import saveData
    
if __name__ == '__main__':
    bTrain = False
    if bTrain:    
       # trainingdata, validationdata, testdata = mnist_loader.load_data_wrapper()
        trainingdata, testdata = mnist_loader.load_data_wrapper_my()
        print 'The sum of trainingdata is: ',len(trainingdata)
        print 'The sum of testdata is: ',len(testdata)
        #net = network.Network([784,30,10])
        net = network.Network([1024,30,10])
        print 'The num of neural :',net.sizes
        print 'Start training...'
        #net.SGD(trainingdata, 30, 10, 3.0, test_data=testdata)
        net.SGD(trainingdata, 40, 10, 1.5, test_data=testdata)
        print 'Training complete!'
        saveData.storeData([net.weights, net.biases],'AnnTrainedParas.txt')
        print 'Saving paras complete! \nThe form is list[weights, biases].'
    else:
        #net2 = network.Network([784,30,10])
        net2 = network.Network([1024,30,10])
        print 'before:',net2.biases
        paras = saveData.grabData('AnnTrainedParas.txt')
        net2.weights = paras[0]
        net2.biases = paras[1]
        print 'after:',net2.biases
        print "各层偏置的维数："
        for a in net2.biases:
            print a.shape
        print "各层权值的维数："
        for b in net2.weights:
            print b.shape
    
    
