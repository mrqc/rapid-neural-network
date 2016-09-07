import neuron

class Layer(object):
	def __init__(self, countOfNeurons):
		if not hasattr(self, "neurons"):
			self.neurons = [neuron.Neuron(self) for _ in range(0, countOfNeurons)]
		self.outputVector = None
	
	def __str__(self):
		neuronsString = ""
		for neuron in self.neurons:
			neuronsString += str(neuron) + "\n"
		return neuronsString
	
	def transfer(self):
		for neuron in self.neurons:
			neuron.transfer()
		for neuron in self.neurons:
			neuron.activation()
		self.outputVector = [neuron.fire for neuron in self.neurons]

class InputLayer (Layer):
	def __init__(self, countOfNeurons):
		super(InputLayer, self).__init__(countOfNeurons)
		self.nextLayer = None

	def connectNextLayer(self, nextLayer):
		self.nextLayer = nextLayer

class OutputLayer (Layer):
	def __init__(self, countOfNeurons):
		super(OutputLayer, self).__init__(countOfNeurons)
		self.previousLayer = None

	def connectPreviousLayer(self, previousLayer):
		self.previousLayer = previousLayer
		for neuron in self.neurons:
			neuron.setCountOfWeights(len(self.previousLayer.neurons))

class HiddenLayer (InputLayer, OutputLayer):
	def __init__(self, countOfNeurons):
		super(HiddenLayer, self).__init__(countOfNeurons)
