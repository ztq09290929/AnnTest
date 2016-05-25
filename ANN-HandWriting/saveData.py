# coding=utf-8
import pickle
'''存储/加载训练好的网络参数'''
def storeData(inputData,filename):
    fw = open(filename,'w')
    pickle.dump(inputData,fw)
    fw.close()

def grabData(filename):
    fr = open(filename)
    return pickle.load(fr)
