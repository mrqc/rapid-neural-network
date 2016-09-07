import random

class Neuron:
	def __init__(self, countOfWeights = 0):
		self.weights = [random.random() for _ in range(0, countOfWeights)]
	
	def __str__(self):
		return str(self.weights)
