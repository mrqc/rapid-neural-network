import neuron

class Layer(object):
	def __init__(self, countOfNeurons, net):
		if not hasattr(self, "net"):
			self.net = net
		if not hasattr(self, "neurons"):
			self.neurons = [neuron.Neuron(self) for _ in range(0, countOfNeurons)]
	
	def __str__(self):
		neuronsString = ""
		for neuron in self.neurons:
			neuronsString += str(neuron) + "\n"
		return neuronsString
	
	def getActivationVector(self):
		return [neuron.activationValue for neuron in self.neurons]
	
	def getEnergyVector(self):
		return [neuron.energy for neuron in self.neurons]
	
class InputLayer (Layer):
	def __init__(self, countOfNeurons, net):
		super(InputLayer, self).__init__(countOfNeurons, net)
		self.nextLayer = None

	def connectNextLayer(self, nextLayer):
		self.nextLayer = nextLayer

	def setEnergyVector(self, energyVector):
		if len(energyVector) != len(self.neurons):
			raise Exception('length of energy vector and neurons count are different')
		for index in range(0, len(self.neurons)):
			self.neurons[index].energy = energyVector[index]
		for neuron in self.neurons:
			neuron.activation()
	
class OutputLayer (Layer):
	def __init__(self, countOfNeurons, net):
		super(OutputLayer, self).__init__(countOfNeurons, net)
		self.previousLayer = None

	def connectPreviousLayer(self, previousLayer):
		self.previousLayer = previousLayer
		for neuron in self.neurons:
			neuron.setCountOfWeights(len(self.previousLayer.neurons))
	
	def backpropagate(self, targetOutputVector):
		for neuron in self.neurons:
			neuron.backpropagate(targetOutputVector)

	def transfer(self):
		for neuron in self.neurons:
			neuron.transfer()
		for neuron in self.neurons:
			neuron.activation()

class HiddenLayer (InputLayer, OutputLayer):
	def __init__(self, countOfNeurons, net):
		super(HiddenLayer, self).__init__(countOfNeurons, net)
