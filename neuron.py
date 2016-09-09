import random
import math
import sys
import layer

class Neuron:
	def __init__(self, layer):
		self.weights = None
		self.trainedWeights = None
		self.energy = 0
		self.layer = layer
		self.threshold = -0.5
		self.activationValue = 0
	
	def setCountOfWeights(self, countOfWeights):
		self.weights = [random.uniform(-1, 1) for _ in range(0, countOfWeights)]
	
	def __str__(self):
		return str(self.weights)

	def transfer(self):
		inputVector = self.layer.previousLayer.getActivationVector()
		if len(inputVector) != len(self.weights):
			raise Exception('length of input vector and weights are different')
		for index in range(0, len(inputVector)):
			self.energy += inputVector[index] * self.weights[index]
	
	def activation(self):
		try:
			self.activationValue = 1 / (1 + math.exp(-self.energy)) # sigmoid function
		except OverflowError:
			self.activationValue = 1

	def activationGradient(self):
		return self.activationValue * (1 - self.activationValue)

	def updateWeights(self):
		self.weights = self.trainedWeights
	
	def error(self, targetActivationValue):
		return 0.5 * (targetActivationValue - self.activationValue) ** 2

	def errorGradient(self, targetActivationValue):
		return self.activationValue - targetActivationValue
	
	def backpropagate(self, errorTotal, targetActivationValue):
		learningRate = 0.002
		self.trainedWeights = [0 for _ in range(0, len(self.weights)]
		for index in range(0, len(self.weights)):
			self.trainedWeights[index] = self.weights[index] - learningRate * self.errorGradient(targetActivationValue) * self.activationGradient() * self.activationValue
