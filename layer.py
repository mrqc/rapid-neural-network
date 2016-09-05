import perceptron

class Layer:
	def __init__(self):
		self.perceptrons = []
	
	def __str__(self):
		perceptronsString = ""
		for perceptron in self.perceptrons:
			perceptronsString += str(perceptron) + "\n"
		return perceptronsString

class InputLayer (Layer):
	def __init__(self, countOfPerceptrons):
		self.perceptrons = [perceptron.Perceptron() for _ in range(0, countOfPerceptrons)]

class HiddenLayer (Layer):
	def __init__(self, countOfPerceptrons, previousLayer = None):
		countOfWeightsPerPerceptron = len(previousLayer.perceptrons)
		self.perceptrons = [perceptron.Perceptron(countOfWeightsPerPerceptron) for _ in range(0, countOfPerceptrons)]

class OutputLayer (HiddenLayer):
	pass

