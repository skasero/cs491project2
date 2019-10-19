# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-10-17 18:15:20
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-10-19 13:30:18
import numpy as np
from clustering import *

data = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])


test_data = np.array([ [1,0], [7,4], [9,6], [2,1], [4,8], [0,3], [13,5], [6,8], 
						[7,3], [3,6], [2,1], [8,3], [10,2], [3,5], [5,1], [1,9],
					    [10,3], [4,1], [6,6], [2,2]    ])

test = K_Means_better(test_data ,3)
print(test)
# C_k = K_Means_better(data, 3)
# print(test_data[1])

#--------Plotting/Write-up Code --------------#

#################
#				#
#   Write-up 1  #
#				#
#################

# clusters_2 = [ [], [] ]
# C_2 = K_Means(test_data,2)
# for i in range(0, len(data)):
# 			best_info = [-1,float("inf")]
# 			for j in range(0,len(C_2)):
# 				distance = np.linalg.norm(C_2[j] - test_data[i])
# 				if distance < best_info[1]:
# 					best_info = [j,distance]
# 			clusters_2[best_info[0]].append(test_data[i])
# print(clusters_2)



clusters_3 = [[], [], []]
