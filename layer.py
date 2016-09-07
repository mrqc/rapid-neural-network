import neuron

class Layer:
	def __init__(self):
		self.neurons = []
	
	def __str__(self):
		neuronsString = ""
		for neuron in self.neurons:
			neuronsString += str(neuron) + "\n"
		return neuronsString

class InputLayer (Layer):
	def __init__(self, countOfNeurons):
		self.neurons = [neuron.Neuron() for _ in range(0, countOfNeurons)]

class HiddenLayer (Layer):
	def __init__(self, countOfNeurons, previousLayer = None):
		countOfWeightsPerNeuron = len(previousLayer.neurons)
		self.neurons = [neuron.Neuron(countOfWeightsPerNeuron) for _ in range(0, countOfNeurons)]

class OutputLayer (HiddenLayer):
	pass

