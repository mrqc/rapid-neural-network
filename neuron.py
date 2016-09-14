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
		self.activationValue = 1
	
	def setCountOfWeights(self, countOfWeights):
		self.weights = [random.uniform(-1, 1) for _ in range(0, countOfWeights)]
	
	def __str__(self):
		return "Weights: " + str(self.weights) + "\nActivation: " + str(self.activationValue) + "\nEnergy: " + str(self.energy)

	def transfer(self):
		self.energy = 0
		inputVector = self.layer.previousLayer.getActivationVector()
		if len(inputVector) != len(self.weights):
			raise Exception('length of input vector and weights are different')
		for index in range(0, len(inputVector)):
			self.energy += inputVector[index] * self.weights[index]
	
	def activation(self): # sigmoid function
		try:
			self.activationValue = 1 / (1 + math.exp(-self.energy))
		except OverflowError:
			self.activationValue = 0

	def activationGradient(self): # d out / d net out = out * (1 - out)
		return self.activationValue * (1 - self.activationValue)

	def updateWeights(self):
		self.weights = self.trainedWeights
	
	def error(self, targetActivationValue): # E = 1/2 * (target - output) ^ 2 mean squared error
		return 0.5 * (targetActivationValue - self.activationValue) ** 2

	def errorGradient(self, targetActivationVector): # d E_total / d out = out - target
		if isinstance(self.layer, layer.HiddenLayer):
			sum = 0
			for index in range(0, len(self.layer.nextLayer.neurons) - 1):
				sum += self.layer.nextLayer.neurons[index].errorGradient(targetActivationVector) * self.layer.nextLayer.neurons[index].activationGradient() * self.layer.nextLayer.neurons[index].weights[self.getIndexInLayer()]
			return sum
		elif isinstance(self.layer, layer.OutputLayer):
			return self.activationValue - targetActivationVector[self.getIndexInLayer()]
	
	def getIndexInLayer(self):
		return self.layer.neurons.index(self)

	def backpropagate(self, errorTotal, targetActivationVector):
		self.trainedWeights = [0 for _ in range(0, len(self.weights))]
		for index in range(0, len(self.weights) - 1):
			self.trainedWeights[index] = self.weights[index] - self.layer.net.learningRate * self.errorGradient(targetActivationVector) * self.activationGradient() * self.layer.previousLayer.neurons[index].activationValue
		self.trainedWeights[len(self.weights) - 1] = self.weights[len(self.weights) - 1] - self.layer.net.learningRateBias * self.errorGradient(targetActivationVector) * self.activationGradient()
