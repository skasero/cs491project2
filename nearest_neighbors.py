#!/usr/bin/python3
import numpy as np

# KNN_test
# returns the accuracy of the model when checked against test data. 
# K is a hyperparameter used to determine how many nearest-neighbors to check against.
def KNN_test(X_train,Y_train,X_test,Y_test,K):
    totalCorrect = 0
    totalCount = len(Y_test)

    # Go through all samples in the test data
    for i in range(0, len(Y_test)):
        neighbors = []
        # Go through all samples in the training data
        for j in range(0,len(Y_train)):
            # Calculate the L2-norm 
            neighbors.append([j,np.linalg.norm(X_test[i]-X_train[j])])
        neighbors = np.array(neighbors) # Needed to convert list to np.array for sorting
        neighbors = neighbors[np.argsort(neighbors[:,1])] # Sort on column 1

        # This is going to add up the label values from the training data which will be used to check if the model was correct
        sum = 0
        for k in range(0,K):
            index = int(neighbors[k,0])
            sum += Y_train[index,0]
        
        # Handle for the case where the sum is 0 on even K values. 
        if(sum != 0):
            sum /= abs(sum) # this is fix a value such as -3 to a -1 value

        if(sum == Y_test[i,0]):
            totalCorrect += 1
        # print("{}  {}".format(i,sum))
    return totalCorrect/totalCount

# choose_K
# will run through all of possible "K" values to find the best "K" based on accuracy
def choose_K(X_train,Y_train,X_val,Y_val):
    bestK = (0,0)
    for i in range(1,len(Y_train)+1):
        acc = KNN_test(X_train,Y_train,X_val,Y_val,i)
        if(acc > bestK[1]):
            bestK = (i,acc)
    return bestK[0]

