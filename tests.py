import unittest
from activelearning import *

class Test (unittest.TestCase) :

	def test_linear_scale(self) :

		X = np.random.randint(0,100, (10,5))
		#print(X)
		X = linear_scale(X)
		#print(X)
		p1 = np.percentile(X, 1, interpolation='nearest', axis=0)
		p5 = np.percentile(X, 5, interpolation='nearest', axis=0)
		p95 = np.percentile(X, 95, interpolation='nearest', axis=0)
		p99 = np.percentile(X, 99, interpolation='nearest', axis=0)
		
		self.assertTrue((p1 <= 0).all())
		self.assertTrue((p5 == 0).all())
		self.assertTrue((p95 == 1).all())
		self.assertTrue((p99 >= 0).all())

		X = np.random.rand(100,100)/2
		
		X = linear_scale(X)
		p1 = np.percentile(X, 1, interpolation='nearest', axis=0)
		p5 = np.percentile(X, 5, interpolation='nearest', axis=0)
		p95 = np.percentile(X, 95, interpolation='nearest', axis=0)
		p99 = np.percentile(X, 99, interpolation='nearest', axis=0)
		
		self.assertTrue((p1 <= 0).all())
		self.assertTrue((p5 == 0).all())
		self.assertTrue((p95 == 1).all())
		self.assertTrue((p99 >= 0).all())

	def test_dist_to_closest(self) :
		# 5 samples
		a = [1, 	3]
		b = [0, 	4]
		c = [2, 	5]
		d = [-1, 	2]
		e = [-1, 	2]
		samples = np.array([a, b, c, d, e])

		dist = distance_to_closest(samples)
		self.assertEqual(dist[0], rbf_kernel([a], [b])[0][0]) # closest to a
		self.assertEqual(dist[1], rbf_kernel([b], [a])[0][0]) # closest to b
		self.assertEqual(dist[2], rbf_kernel([c], [b])[0][0]) # closest to c
		self.assertEqual(dist[3], rbf_kernel([d], [e])[0][0]) # closest to d
		self.assertEqual(dist[4], rbf_kernel([d], [e])[0][0]) # closest to e


	def test_average_dist(self) :
		# 3 samples
		a = [1, 	3]
		b = [0, 	4]
		c = [2, 	5]
		samples = np.array([a, b, c])
		dist = average_distance(samples)

		avg_a = (rbf_kernel([a], [b])[0][0] + rbf_kernel([a], [c])[0][0])/2
		avg_b = (rbf_kernel([b], [a])[0][0] + rbf_kernel([b], [c])[0][0])/2
		avg_c = (rbf_kernel([c], [a])[0][0] + rbf_kernel([c], [b])[0][0])/2
		
		self.assertAlmostEqual(dist[0],avg_a)
		self.assertAlmostEqual(dist[1],avg_b)
		self.assertAlmostEqual(dist[2],avg_c)

	def test_diversity_criterion(self) :
		# 9 samples -> plot them for a better visualization (e.g. with GeoGebra)
		a = [2, 	4]
		b = [4, 	1]
		b_bis = [4, 	1]
		c = [-2, 	1]
		d = [-1, 	5]
		e = [1.6, 	3.6]
		f = [3, 	1]
		g = [1, 	2]
		h = [9, 	5]
		i = [49, 	4] # This sample is not sent to the diversity filter
		samples = np.array([a, b, b_bis, c, d, e, f, g, h, i])

		selected_samples = diversity_filter(samples, np.arange(9), 4)

		self.assertTrue(0 not in selected_samples)
		self.assertTrue((1 in selected_samples) ^ (2 in selected_samples))	# Either b or b_bis is kept
		self.assertTrue(3 in selected_samples)
		self.assertTrue(4 in selected_samples)
		self.assertTrue(5 not in selected_samples)
		self.assertTrue(6 not in selected_samples)
		self.assertTrue(7 not in selected_samples)
		self.assertTrue(8 in selected_samples)

		self.assertTrue((samples == np.array([a, b, b_bis, c, d, e, f, g, h, i])).all())	# Check that the original array was not modified


	def test_write_result_file(self) :
		X = np.array([
			[11., 3.5, 4.7],
			[22., 4.5, 6.7]
		])
		header = np.array(['ID', 'attr1','attr2', 'attr3'])
		ID = np.array([1., 2.])
		predictions = np.array([0,1])
		filename = 'unittest_prediction.csv'

		#ID, X_unlabeled, predictions, header, filename

		write_result_file(ID, X, predictions, header, filename)

		result = np.genfromtxt(filename, delimiter=',', dtype=str)

		data = np.array([
			['ID', 'Class', 'attr1','attr2', 'attr3'],
			[1., 0., 11., 3.5, 4.7],
			[2., 1., 22., 4.5, 6.7]
		])

		self.assertTrue((data == result).all())


if __name__ == '__main__' :
	unittest.main()