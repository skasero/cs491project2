  
# -*- coding: utf-8 -*-
# @Author: Chris Peterson
# @Date:   2019-10-17 16:52:38
# @Last Modified by:   Chris Peterson
# @Last Modified time: 2019-10-19 14:02:39
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
		center = X[i].tolist()
		if center not in cluster_centers:
			cluster_centers.append(center)
	#Flag to determine if main algorithm has converged
	converged = False

	# Array to store cluster from previous run to determine if convergence has happened.
	clusters_previous = []

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
				distance = np.linalg.norm(np.subtract(cluster_centers[j],X[i])).tolist()
				if distance < best_info[1]:
					best_info = [j,distance]
			clusters[best_info[0]].append(X[i].tolist())

		# Check if anything has changed from last iteration
		if np.array_equal(clusters, clusters_previous):
			converged = True
		else:
			# Recompute cluster centers
			for i in range(0, len(cluster_centers)):
				if not np.array_equal(clusters[i], []):
					cluster_centers[i] = np.mean(clusters[i], axis=0, dtype=float)
				# else: Points could be reassigned to the empty cluster later. Lets keep the center
					
			clusters_previous = copy.deepcopy(clusters)
			clusters = [[] for i in range(0,K)]
	cluster_centers = np.array(cluster_centers)
	cluster_centers = cluster_centers[np.argsort(cluster_centers[:,0])] 
	return cluster_centers

#
# K_Means_better(X,K)
# @param X A numpy array of features, indexed by sample
# @param K An integer representing the number of cluster centers desired
# @return cluster_centers A numpy array of (float) cluster centers that was found to be the best over many iterations
# 						of k_means(X,K)
#			
def K_Means_better(X,K):
	models = []
	model_votes = []
	cluster_centers = []
	MIN_ITERATIONS = 500
	MAX_ITERATIONS = 1000
	iteration_number = 0
	found_majority = False
	sigma = .005
	best_model = []

	while iteration_number < MAX_ITERATIONS and not found_majority:
		current_model = K_Means(X,K).tolist()
		closest_match = [[], float("inf")]

		if current_model in models:
			index = models.index(current_model)
			model_votes[index] = model_votes[index] + 1
			if (model_votes[index]/(iteration_number+1) > 0.5) and iteration_number > MIN_ITERATIONS:
				found_majority = True
				best_model = current_model
		else:
			for model in models:
				difference = np.linalg.norm(np.subtract(model,current_model)).tolist()
				if difference < closest_match[1]:
					closest_match = [model, difference]
			if closest_match[1] < sigma:
				index = models.index(closest_match[0])
				model_votes[index] = model_votes[index] + 1
				if (model_votes[index]/(iteration_number+1) > 0.5) and iteration_number > MIN_ITERATIONS :
					found_majority = True
			else:
				models.append(current_model)
				model_votes.append(1)

		iteration_number = iteration_number + 1

	if not found_majority:
		best_index = model_votes.index(max(model_votes))
		best_model = models[best_index]

	return best_model