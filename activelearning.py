#!/usr/bin/env python
# encoding: utf-8
#******************************************************************
#*
#* MODULE:		activelearning
#*
#* AUTHOR(S)	Lucas Lefèvre
#*
#* PURPOSE:		Remote image classification
#*
#* COPYRIGHT:	(C) 2017 Lucas Lefèvre
#*				Bruxelles, Belgique
#*
#******************************************************************

#%module
#% description: Remote image classification
#%end
#%option
#% key: training_set
#% type: string
#% gisprompt: file, dsn
#% answer:
#% description: Training set (csv format)
#% required: yes
#%end
#%option
#% key: test_set
#% type: string
#% gisprompt: file, dsn
#% answer:
#% description: Test set (csv format)
#% required: yes
#%end
#%option
#% key: unlabeled_set
#% type: string
#% gisprompt: file, dsn
#% answer:
#% description: Unlabeled samples (csv format)
#% required: yes
#%end
#%option
#% key: learning_steps
#% type: integer
#% description: Number of samples to label at each iteration
#% required: no
#%end
#%option
#% key: diversity_select_from
#% type: integer
#% description: Number of samples to select (based on uncertainty criterion) before applying the diversity criterion.
#% required: no
#%end
#%option
#% key: diversity_lambda
#% type: double
#% description: Lambda parameter used in the diversity heuristic
#% required: no
#%end
#%option
#% key: c_parameter
#% type: double
#% description: Penalty parameter C of the error term
#% required: no
#%end
#%option
#% key: gamma_parameter
#% type: double
#% description: Kernel coefficient
#% required: no
#%end


"""
	learning_steps = 5			# Number of samples to label at each iteration
	learning_iterations = 50	# Number of iterations for the active learning process
	diversity_lambda = 0.25		# Lambda parameter used in the diversity heuristic
	diversity_select_from = 15 	# Number of samples to select (based on uncertainty criterion) before applying the diversity criterion. Must be at least greater or equal to [LEARNING][steps]
	test_trials = 80			# Number of trials for computing the average scores
	test_start_with = 60		# Number of labeled samples to use for the first iteration
	test_unlabeled_pool = 800	# Number of unlabeled samples

"""

import grass as grass
from grass.script.core import gisenv
from grass.pygrass import raster

import numpy as np 
from sklearn import svm
from sklearn import preprocessing
import ConfigParser as configparser

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel


import time
import sys

import matplotlib.pyplot as plt


def auto_learning(X, y, ID, steps, iterations, sample_selection) :
	"""
	Functiun for testing purposes only.  We already know the label of every sample (X, y).
	Automaticaly get the labels from parameter y at each learning iteration.
	"""

	m = test_start_with	#number of initial training examples
	p = test_unlabeled_pool	#pool of unlabeled samples


	# Training set
	X_train = X[:m,:]
	y_train = y[:m]
	ID_train = ID[:m]

	# Pool of unlabeled data
	X_pool = X[m:p,:]
	y_pool = y[m:p]
	ID_pool = ID[m:p]

	# Test set
	X_test = X[p:,:]
	y_test = y[p:]
	ID_test = ID[p:]

	iterations_score = np.empty(iterations)

	for i in range(0, iterations) :

		if(X_pool.size == 0) :
			raise Exception("Pool of unlabeled samples empty")

		classifier = train(X_train, y_train)
		score = classifier.score(X_test, y_test)
		#print('({}) Score : {} | {} labeled samples | {} samples in the pool'.format(i, score, X_train.shape[0],X_pool.shape[0]))
		
		iterations_score[i] = score
		samples_to_label = sample_selection(X_pool, steps, classifier)
		
		# Add new labeled samples to the training set
		X_train = np.concatenate((X_train,  X_pool[samples_to_label]), axis=0)
		y_train = np.concatenate((y_train,  y_pool[samples_to_label]), axis=0)

		# Delete new labeled samples from the pool
		X_pool = np.delete(X_pool, samples_to_label, axis=0)
		y_pool = np.delete(y_pool, samples_to_label, axis=0)

	return iterations_score

def testing() :
	
	steps = learning_steps # Number of sample to label at each iteration
	iterations = learning_iterations # Number of iteration in the active learning process
	repeated = test_trials #number of runs for the average score
	
	score_active = np.empty([repeated, iterations])
	score_active_diversity = np.empty([repeated, iterations])
	score_random = np.empty([repeated, iterations])
	
	
	
	for i in range(repeated) :
		X, y, ID = load_data('training_sample.csv')
		scores = learning(X, y, ID, steps, iterations, random_sample_selection)
		print("Random learning ({})".format(i))
		score_random[i] = scores

	for i in range(repeated) :
		X, y, ID = load_data('training_sample.csv')
		scores = learning(X, y, ID, steps, iterations, active_diversity_sample_selection)
		print("Acitve learning with diversity criterion ({})".format(i))
		score_active_diversity[i] = scores
	
	for i in range(repeated) :
		X, y, ID = load_data('training_sample.csv')
		scores = learning(X, y, ID, steps, iterations, active_sample_selection)
		print("Active learning without diversity criterion ({})".format(i))
		score_active[i] = scores

	

	start = test_start_with

	draw_graph(score_active, score_active_diversity, score_random, np.arange(start, start+iterations*steps, steps))
	
	

	#plt.plot(score_active, [x for x in range(0, len(score_active))])
	"""
	parameters = {
		'kernel': ('linear', 'rbf'),
		'C':[ 10, 5, 1, 1e-3],
		'gamma' : np.logspace(-2, 2, 5),
		'decision_function_shape':('ovo', 'ovr')
	}

	svr = svm.SVC()
	clf = GridSearchCV(svr, parameters, n_jobs=4, verbose=0)

	t0 = time.time()
	clf.fit(X[:100], y[:100])

	print("Parameters searched in " + str((time.time() - t0)) + " second(s)")


	print(clf.best_score_)
	print(clf.best_params_)
	"""


def draw_graph(score_active, score_active_diversity, score_random, examples) :
	mu1 = score_active.mean(axis=0)
	sigma1 = score_active.std(axis=0)

	mu2 = score_active_diversity.mean(axis=0)

	mu3 = score_random.mean(axis=0)
	sigma3 = score_random.std(axis=0)

	fig, ax = plt.subplots(1)
	
	
	ax.plot(examples, mu1, lw=2, label='Active learning without diversity criterion', color='blue')
	#ax.fill_between(examples, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)
	ax.plot(examples, mu2, lw=2, label='Active learning with diversity criterion', color='green')
	ax.plot(examples, mu3, lw=2, label='Random learning', color='yellow')
	#ax.fill_between(examples, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
	ax.legend(loc='upper left')
	ax.set_ylabel('Score')
	ax.set_xlabel('Number of training examples')

	ax.grid(25)
	plt.show()


def load_data(file_path, labeled=False) :

	data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
	#np.random.shuffle(data)

	ID = data[:,0] #get only row 0
	if labeled :
		y = data[:,1] #get only row 1
		X = data[:,2:] #remove ID and label
	else :
		y = []
		X = data[:,1:] #remove ID

	X = preprocessing.scale(X)
	#X = linear_scale(X)
	return X, ID, y

def linear_scale(data) :
	"""
		Linearly scale values : 5th percentile to 0 and 95th percentile to 1 percentile
	"""
	p5 = np.percentile(data, 5, axis=0, interpolation='nearest')[np.newaxis] 	# 5th percentiles as a 2D array (-> newaxis)
	p95 = np.percentile(data, 95, axis=0, interpolation='nearest')[np.newaxis]	# 95th percentiles as a 2D array (-> newaxis)
	
	return (data-p5)/(p95-p5)

def train(X, y, c_parameter, gamma_parameter) :
	classifier = svm.SVC(kernel='rbf', C=c_parameter, gamma=gamma_parameter, probability=True,decision_function_shape='ovo', random_state=1938475632)
	t0 = time.time()
	classifier.fit(X, y)

	return classifier

def random_sample_selection(X_unlabled, nbr, classifier=None) :
	return np.random.choice(X_unlabled.shape[0], nbr)

def active_sample_selection(X_unlabled, nbr, classifier) :
	"""
		Select a number of samples to label based only on uncertainety 
	"""
	return uncertainty_filter(X_unlabled, nbr, classifier)

def active_diversity_sample_selection(X_unlabled, nbr, classifier) :
	"""
		Select a number of samples to label based on uncertainety and diversity
	"""
	
	batch_size = diversity_select_from	# Number of samples to select with the uncertainty criterion

	uncertain_samples_index = uncertainty_filter(X_unlabled, batch_size, classifier)	# Take twice as many samples as needed
	uncertain_samples = X_unlabled[uncertain_samples_index]
	
	return diversity_filter(uncertain_samples, uncertain_samples_index, nbr)

def uncertainty_filter(samples, nbr, classifier) : 
	"""
		Keep only 'nbr' samples based on an uncertainty criterion		
		Return the indexes of samples to keep
	"""
	NBR_NEW_SAMPLE = nbr
	decision_function = classifier.predict_proba(samples)

	# Check if the number of samples to return is not 
	# bigger than the total number of samples
	if (nbr >= samples.shape[0]) :
		NBR_NEW_SAMPLE = samples.shape[0] - 1
	

	# Get the max distance to each class hyperplane for each example
	max_index = np.argmax(decision_function[:,:], axis=1)
	max_values = decision_function[np.arange(len(decision_function)), max_index]

	# Make the max values very small.
	# The max value is now the second best
	decision_function[np.arange(len(decision_function)), max_index] = np.NINF 
	
	# Get the second max distance to each class to hyperplane for each example
	second_max_index = np.argmax(decision_function[:,:], axis=1)
	second_max_values = decision_function[np.arange(len(decision_function)), second_max_index]

	# "Functionnal margin" for multiclass classifiers for each sample
	f_MC = max_values - second_max_values

	
	selected_sample_index = np.argpartition(f_MC, NBR_NEW_SAMPLE)[:NBR_NEW_SAMPLE]

	return selected_sample_index

def diversity_filter(samples, uncertain_samples_index, nbr) :
	"""
		Keep only 'nbr' samples based on a diversity criterion (bruzzone2009 : Active Learning For Classification Of Remote Sensing Images)
		Return the indexes of samples to keep
	"""
	L = diversity_lambda
	m = samples.shape[0]	# Number of samples

	samples_cpy = np.empty(samples.shape)
	samples_cpy[:] = samples

	selected_sample_index = uncertain_samples_index	# At the begining, take all samples

	while (selected_sample_index.shape[0] > nbr) :

		dist_to_closest = distance_to_closest(samples_cpy)
		average_dist = average_distance(samples_cpy)
		discard = np.argmax(L*dist_to_closest + ((1-L) * (1/m) * average_dist))
		
		selected_sample_index = np.delete(selected_sample_index, discard)	# Remove the sample to discard
		samples_cpy = np.delete(samples_cpy, discard, axis=0)
	
	return selected_sample_index

def distance_to_closest(samples, distances=None) :
	"""
		For each sample, computes the distance to its closest neighbour
		return size : #samples x 1
	"""
	dist_with_samples = rbf_kernel(samples, samples) # Distance between each samples (symetric matrix)
	np.fill_diagonal(dist_with_samples, np.NINF) # Do not take into acount the distance between a sample and itself (values on the diagonal)
	dist_with_closest = dist_with_samples.max(axis=0) # For each sample, the distance to the closest other sample
	
	return dist_with_closest


def average_distance(samples) :
	"""
		For each sample, computes the average distance to all other samples  
		return size : #samples x 1
	"""
	samples = np.asarray(samples)
	nbr_samples = samples.shape[0]
	dist_with_samples = rbf_kernel(samples, samples)
	average_dist = (dist_with_samples.sum(axis=1) - 1)/(nbr_samples-1)	# Remove dist to itself (=1)
	
	return average_dist

def learning(X_train, y_train, X_test, y_test, X_unlabeled, ID_unlabeled, steps, sample_selection) :



	if(X_unlabeled.size == 0) :
		raise Exception("Pool of unlabeled samples empty")

	c_parameter, gamma_parameter = SVM_parameters(options['c_parameter'], options['gamma_parameter'], X_train, y_train)
	print('Parameter used : C={}, gamma={}, lambda={}'.format(c_parameter, gamma_parameter, diversity_lambda))
	
	classifier = train(X_train, y_train, c_parameter, gamma_parameter)
	score = classifier.score(X_test, y_test)
	
	
	samples_to_label = sample_selection(X_unlabeled, steps, classifier)

	return ID_unlabeled[samples_to_label], score

def SVM_parameters(c, gamma, X_train, y_train) :

	parameters = {}
	if c == '' :
		parameters['C'] = [ 10, 5, 1, 1e-3]
	if gamma == '' :
		parameters['gamma'] = np.logspace(-2, 2, 5)
	
	if parameters != {} :
		svr = svm.SVC()
		clf = GridSearchCV(svr, parameters, verbose=0)
		clf.fit(X_train, y_train)

	if c == '' :
		c = clf.best_params_['C']
	if gamma == '' :
		gamma = clf.best_params_['gamma']
	return float(c), float(gamma)

def main(options, flags) :
	print("Active Learning")

	X_train, ID_train, y_train = load_data(options['training_set'], labeled = True)
	X_test, ID_test, y_test = load_data(options['test_set'], labeled = True)
	X_unlabeled, ID_unlabeled, y_unlabeled = load_data(options['unlabeled_set'])
	

	samples_to_label_IDs, score = learning(X_train, y_train, X_test, y_test, X_unlabeled, ID_unlabeled, learning_steps, active_diversity_sample_selection)
	
	print('Training set : {}'.format(X_train.shape[0]))
	print('Test set : {}'.format(X_test.shape[0]))
	print('Unlabeled set : {}'.format(X_unlabeled.shape[0]))
	print('Score : {}'.format(score))
	print('--------------------------')
	print('Label the following samples to improve the score :')
	print(samples_to_label_IDs)
	print('--------------------------')



	"""
	segments = raster.RasterRow('geobia')
	print(segments.exist())
	segments.open()
	print(segments.is_open())
	print(segments.info)
	print(segments[30546:30566])
	
	segments.close()
 	import wx
 	from gui import MainWindow


 	app = wx.App(False)
 	frame = MainWindow(None, "Active Learning")
 	app.MainLoop()
 	"""

if __name__ == '__main__' :
	options, flags = grass.script.parser()

	

	# Some global variables (the user will be able to choose the value)
	learning_steps = int(options['learning_steps']) if options['learning_steps'] != '' else 5					# Number of samples to label at each iteration
	diversity_lambda = float(options['diversity_lambda']) if options['diversity_lambda'] != '' else 0.25		# Lambda parameter used in the diversity heuristic
	diversity_select_from = int(options['diversity_select_from']) if options['diversity_select_from'] != '' else 15 	# Number of samples to select (based on uncertainty criterion) before applying the diversity criterion. Must be at least greater or equal to [LEARNING][steps]

	# Only for testing purposes
	test_trials = 80			# Number of trials for computing the average scores
	test_start_with = 60		# Number of labeled samples to use for the first iteration
	test_unlabeled_pool = 800	# Number of unlabeled samples
	learning_iterations = 50	# Number of iterations for the active learning process
	
	main(options, flags)
