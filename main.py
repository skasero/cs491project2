import numpy as np
import nearest_neighbors as nn

if __name__ == "__main__":
    knn1X = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
    knn1Y = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
    knn2X = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
    knn2Y = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

    acc = nn.KNN_test(knn1X,knn1Y,knn2X,knn2Y,3)
    print(acc)
    print(nn.choose_K(knn1X,knn1Y,knn2X,knn2Y))