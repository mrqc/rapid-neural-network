import random
import math

class Neuron:
	def __init__(self, layer):
		self.weights = None
		self.energy = 0
		self.layer = layer
		self.fire = 0
		self.threshold = -0.5
		self.activationValue = 0
	
	def setCountOfWeights(self, countOfWeights):
		self.weights = [random.uniform(-1, 1) for _ in range(0, countOfWeights)]
	
	def __str__(self):
		return str(self.weights)

	def transfer(self):
		inputVector = self.layer.previousLayer.outputVector
		if len(inputVector) != len(self.weights):
			raise Exception('length of input vector and weights are different')
		for index in range(0, len(inputVector)):
			self.energy += inputVector[index] * self.weights[index]
	
	def activation(self):
		a = 5
		v = self.energy
		self.activationValue = 1 / (1 + math.exp(-(a * v)))
		if self.activationValue + self.threshold >= 0:
			self.fire = 1
		else:
			self.fire = 0
