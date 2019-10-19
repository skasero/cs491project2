#!/usr/bin/python3
import numpy as np

def perceptron_train(X,Y):
    # This is used to fix arrays that have 0 as their labels instead of -1
    for i in range(0,len(Y)):
        if(Y[i] != 1):
            Y[i] = -1
    index = 0
    countToEpoch = 0
    epoch = len(Y)
    w = [0] * len(X[0]) # Creating a list of [0] * number of features
    b = 0 # Starts as 0
    
    while(countToEpoch != epoch-1):
        a = (np.dot(w,X[index])+b)*Y[index]
        if(a <= 0):
            w += X[index]*Y[index]
            b += Y[index]
            countToEpoch = 0
        else:
            countToEpoch += 1
        index += 1
        index %= epoch
    # print("w: {} b: {}".format(w,b))
    return [w.tolist(),b.tolist()]

def perceptron_test(X_test, Y_test, w, b):
    totalCorrect = 0
    totalCount = len(Y_test)

    for i in range(0,len(Y_test)):
        y = np.dot(w,X_test[i]) + b
        y = y[0]
        if(y != 0):
            y /= abs(y)
        
        if(y == Y_test[i]):
            totalCorrect += 1
    return totalCorrect/totalCount 