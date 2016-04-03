import numpy as np
import cv2
#encoding: utf - 8

def salt(img,n):
    for i in range(n):
        i = int(np.random.random()*img.shape[0])
        j = int(np.random.random()*img.shape[1])
        if img.ndim == 2:
            img[i,j] = 0
        elif img.ndim ==3:
            img[i,j,0] = 0
            img[i,j,1] = 0
            img[i,j,2] = 0
    return img

if __name__ == '__main__':#如果本文档当作程序执行，该代码会被运行，如果当作模块导入则被运行
    img = cv2.imread("C:\\Users\\loves_000\\Desktop\\制作相册、\\14.jpg")
    saltImg = salt(img,700)
    cv2.imshow('salt',saltImg)
    cv2.waitKey(0)
