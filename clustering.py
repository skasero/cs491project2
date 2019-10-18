# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-10-17 16:52:38
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-10-17 18:22:45
import numpy as np
import random as rand

def K_Means(X,K):
	# set up an array for cluster centers (grabs a random sample's features)
	initial_centers = []
	rand.seed()
	while len(initial_centers) < K:
		i = rand.randint(0,len(X) - 1)
		if i not in initial_centers:
			initial_centers.append(X[i])


	#Flag for if algorithm has converged
	converged = False
	#Empty clusters array with empty tuples
	clusters = [[] for i in range(0,K)]	
	print(clusters)	

	while not converged:
		for i in range(0, len(X)):
			distances = (-1,10000000000000)
			for j in range(0,len(initial_centers)):
				distance = np.linalg.norm(initial_centers[j] - X[i])
				if distance < distances[1]:
					distances = (j,distance)
			clusters[distances[0]].append(i)
		# Recompute cluster centers
		print(clusters)
		for c in range(0,len(initial_centers)):
			# print(clusters)
			exit()
