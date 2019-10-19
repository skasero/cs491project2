import numpy as np
import nearest_neighbors as nn
import perceptron as p
import clustering as c

import matplotlib.pyplot as plt

if __name__ == "__main__":
    knn1X = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
    knn1Y = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
    knn2X = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
    knn2Y = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

    acc = nn.KNN_test(knn2X,knn2Y,knn1X,knn1Y,9)
    #print(acc)
    #print(nn.choose_K(knn2X,knn2Y,knn1X,knn1Y))

    X = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
    Y = np.array([[1], [1], [0], [1], [0], [0], [0]])
    X1 = np.array([[-1,-1],[-1,0],[0,-1],[1,1]])
    Y1 = np.array([[-1],[-1],[-1],[1]])

    XX = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
    YY = np.array([[1], [1], [1], [-1], [-1], [-1]])
    W = p.perceptron_train(X1, Y1)
    test_acc = p.perceptron_test(X, Y, W[0], W[1])
    #print(W)
    #print(test_acc)

    # Plotting graph
    # XPlot = [-2,1,1.5,-2,-1,2]
    # YPlot = [1,1,-.5,-1,-1.5,-2]
    # plt.plot(XPlot,YPlot,'o')
    # x = np.linspace(-2.5,2.5,100)
    # y = -.5*x-.5
    # plt.ylim([-2.5,1.5])
    # plt.xlim([-2.5,2.5])
    # plt.plot(x, y, '-r', label='y=-0.5x-0.5')
    # plt.xlabel('x', color='#1C2833')
    # plt.ylabel('y', color='#1C2833')
    # plt.legend(loc='upper right')
    # plt.grid()
    # plt.savefig('foo.png')
    

    X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
    XCluster = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
    K = 3
    C = c.K_Means_better(XCluster, K)
    print(C)