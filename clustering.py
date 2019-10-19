# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-10-17 16:52:38
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-10-19 00:38:34
import numpy as np
import random as rand
import copy

#
# K_Means(X,K)
# @param X A numpy array of features, indexed by sample
# @param K An integer representing the number of cluster centers desired
# @return cluster_centers A list of (float) cluster centers
#
def K_Means(X,K):

	# set up an array for cluster centers (grabs a random sample's features)
	cluster_centers = []

	# Seed the random number generator for cluster center initialization
	rand.seed()

	# Randomly choose K samples as initial cluster centers
	while len(cluster_centers) < K:
		i = rand.randint(0,len(X) - 1)   # Note: len(X) - 1 Since randint(a,b) generates [a,b]
		if X[i] not in cluster_centers:
			cluster_centers.append(X[i])

	# print(cluster_centers)
	#Flag to determine if main algorithm has converged
	converged = False

	# Array to store cluster from previous run to determine if convergence has happened.
	clusters_previous = np.array([])

	# clusters is a list of the samples (stored by index of X) that belong to a particluar cluster
	# The indicies for clusters is meant to be the same for cluster_centers
	# e.g. clusters[0] is the list for cluster_centers[0], etc
	# To make assignment easier later, we initial the array with K empty lists for this
	clusters = [[] for i in range(0,K)]
	

	#######################
	#                     #
	# Main algorithm loop #
	#                     #
	#######################
	while not converged:
		for i in range(0, len(X)):
			best_info = [-1,float("inf")]
			for j in range(0,len(cluster_centers)):
				distance = np.linalg.norm(cluster_centers[j] - X[i])
				# print("Center: ", cluster_centers[j], "Sample: ", X[i,0], "Distance: ", distance)
				if distance < best_info[1]:
					best_info = [j,distance]
			clusters[best_info[0]].append(i)
			# print("----------------------------------------")
			# print(clusters)
			# print("----------------------------------------")

		#print("Previous clusters: ", clusters_previous)
		# print("Current clusters: ", clusters)	
		# Check if anything has changed from last iteration
		if np.array_equal(clusters, clusters_previous):
			converged = True
		else:
			# Recompute cluster centers
			for i in range(0, len(cluster_centers)):
				if not np.array_equal(clusters[i], []):
					cluster_centers[i] = [np.mean(clusters[i], axis=0, dtype=float)]
				# else: Points could be reassigned to the empty cluster later. Lets keep the center
					
					
			clusters_previous = copy.deepcopy(clusters)
			clusters = [[] for i in range(0,K)]

	return cluster_centers
		
			
