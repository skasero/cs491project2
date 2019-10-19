#!/usr/bin/python3
import numpy as np

# perceptron_train
# returns the weights and bias for training data
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
    
    # Continue going through all samples until an entire epoch where there was no update.
    while(countToEpoch != epoch-1):
        a = (np.dot(w,X[index])+b)*Y[index]
        # We need to update if a <= 0
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

# perceptron_test
# checks how accurate the weights and bias are for the perceptron by testing on test data
def perceptron_test(X_test, Y_test, w, b):
    totalCorrect = 0
    totalCount = len(Y_test)

    for i in range(0,len(Y_test)):
        y = np.dot(w,X_test[i]) + b
        y = y[0]
        if(y != 0):
            y /= abs(y) # this is fix a value such as -3 to a -1 value
        
        if(y == Y_test[i]):
            totalCorrect += 1
    return totalCorrect/totalCount 