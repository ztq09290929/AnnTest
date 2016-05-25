import pickle

def storeData(inputData,filename):
    fw = open(filename,'w')
    pickle.dump(inputData,fw)
    fw.close()

def grabData(filename):
    fr = open(filename)
    return pickle.load(fr)
