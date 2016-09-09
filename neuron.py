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
			a = 5
			self.activationValue = 1 / (1 + math.exp(-(a * self.energy))) # sigmoid function
		except OverflowError:
			self.activationValue = 1
	
	def backpropagate(self, targetOutputVector):
		if len(targetOutputVector) != len(self.layer.getActivationVector()):
			raise Exception('length of golden standard vector and current vector different')
		learningRate = 0.0002
		gradientDescentEpsilon = 0.003
		updatedWeights = [weight for weight in self.weights]
		lastError = self.cost(targetOutputVector, self.layer.getActivationVector())
		newError = lastError + gradientDescentEpsilon
		while math.fabs(newError - lastError) >= gradientDescentEpsilon:
			weightDelta = [0 for _ in range(0, len(self.weights))]
			for index in range(0, len(self.weights)):
				weightDelta[index] = learningRate * self.costDerivative(targetOutputVector, self.layer.getActivationVector())
			for index in range(0, len(self.weights)):
				updatedWeights[index] = updatedWeights[index] - weightDelta[index]
			newError = self.cost(targetOutputVector, self.layer.getActivationVector())
			lastError = newError
		self.trainedWeights = updatedWeights
	
	def updateWeights(self):
		self.weights = self.trainedWeights

	def cost(self, hypothesis, current):
		sum = 0
		for index in range(0, len(hypothesis)):
			sum += (hypothesis[index] - current[index]) ** 2
		return sum / 2

	def costDerivative(self, hypothesis, current):
		sum = 0
		for index in range(0, len(hypothesis)):
			sum += (hypothesis[index] - current[index])
		return sum / 2
