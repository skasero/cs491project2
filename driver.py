# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-10-17 18:15:20
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-10-19 11:31:13
import numpy as np
from clustering import *

data = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
C = K_Means(data,3)

test_data = np.array([ [1,0], [7,4], [9,6], [2,1], [4,8], [0,3], [13,5], [6,8], 
						[7,3], [3,6], [2,1], [8,3], [10,2], [3,5], [5,1], [1,9],
					    [10,3], [4,1], [6,6], [2,2]    ])

