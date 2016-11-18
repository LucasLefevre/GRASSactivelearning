from sklearn import svm
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import numpy as np 
import time

import matplotlib.pyplot as plt

def draw_graph(score_active, score_random, examples) :
	mu1 = score_active.mean(axis=0)
	sigma1 = score_active.std(axis=0)

	mu2 = score_random.mean(axis=0)
	sigma2 = score_random.std(axis=0)

	fig, ax = plt.subplots(1)
	
	
	ax.plot(examples, mu1, lw=2, label='Active learning', color='blue')
	ax.fill_between(examples, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.5)
	ax.plot(examples, mu2, lw=2, label='Random learning', color='yellow')
	ax.fill_between(examples, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
	ax.legend(loc='upper left')
	ax.set_ylabel('Score')
	ax.set_xlabel('Training examples')

	ax.grid()
	plt.show()


def load_data(file_path) :

	data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
	np.random.shuffle(data)

	ID = data[:,0] #get only row 0
	y = data[:,1] #get only row 1
	X = data[:,2:] #remove ID and label
	X = preprocessing.scale(X)

	return X, y, ID

def train(X, y) :
	classifier = svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True,decision_function_shape='ovo')
	t0 = time.time()
	classifier.fit(X, y)
	# print("Model fitted in " + str((time.time() - t0)) + " second(s)")

	return classifier

def random_sample_selection(classifier, X_unlabled, nbr=10) :
	return np.random.choice(X_unlabled.shape[0], nbr)

def active_sample_selection(classifier, X_unlabled, nbr=10) : 

	NBR_NEW_SAMPLE = nbr

	# Check if the number of samples to return is not 
	# bigger than the total number of samples
	if (nbr >= X_unlabled.shape[0]) :
		NBR_NEW_SAMPLE = X_unlabled.shape[0] - 1
	

	decision_function = classifier.predict_proba(X_unlabled)

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



def learning(X, y, ID, steps, iterations, sample_selection) :

	m = 20 #number of training examples
	p =  800#pool of unlabeled samples


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

		classifier = train(X_train, y_train)
		score = classifier.score(X_test, y_test)
		#print('({}) Score : {} | {} labeled samples | {} samples in the pool'.format(i, score, X_train.shape[0],X_pool.shape[0]))
		
		iterations_score[i] = score
		samples_to_label = sample_selection(classifier, X_pool, steps)
		
		# Add new labeled samples to the training set
		X_train = np.concatenate((X_train,  X_pool[samples_to_label]), axis=0)
		y_train = np.concatenate((y_train,  y_pool[samples_to_label]), axis=0)

		# Delete new labeled samples from the pool
		X_pool = np.delete(X_pool, samples_to_label, axis=0)
		y_pool = np.delete(y_pool, samples_to_label, axis=0)

	return iterations_score

def main() :
	
	steps = 5 # Number of sample to label at each iteration
	iterations = 50 # Number of iteration in the active learning process
	repeated = 1
	
	score_active = np.empty([repeated, iterations])
	score_random = np.empty([repeated, iterations])
	

	for i in range(repeated) :
		X, y, ID = load_data('training_sample.csv')
		scores = learning(X, y, ID, steps, iterations, random_sample_selection)
		print("Random learning ({})".format(i))
		score_random[i] = scores

	for i in range(repeated) :
		X, y, ID = load_data('training_sample.csv')
		scores = learning(X, y, ID, steps, iterations, active_sample_selection)
		print("Active learning ({})".format(i))
		score_active[i] = scores

	


	draw_graph(score_active, score_random, np.arange(20, 20+iterations*steps, steps))
	


	#plt.plot(score_active, [x for x in range(0, len(score_active))])
	
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
	


if __name__ == '__main__' :
	main()