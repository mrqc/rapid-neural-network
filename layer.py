import neuron

class Layer(object):
	def __init__(self, countOfNeurons, net):
		if not hasattr(self, "net"):
			self.net = net
		if not hasattr(self, "neurons"):
			self.neurons = [neuron.Neuron(self) for _ in range(0, countOfNeurons + 1)]
	
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
		self.activationVector = None
	
	def connectNextLayer(self, nextLayer):
		self.nextLayer = nextLayer

	def setActivationVector(self, activationVector):
		self.activationVector = list(activationVector)
		self.activationVector.append(1)
		if len(self.activationVector) != len(self.neurons):
			raise Exception('length of energy vector and neurons count are different')
		for index in range(0, len(self.neurons)):
			self.neurons[index].activationValue = self.activationVector[index]

class OutputLayer (Layer):
	def __init__(self, countOfNeurons, net):
		super(OutputLayer, self).__init__(countOfNeurons, net)
		self.previousLayer = None
	
	def connectPreviousLayer(self, previousLayer):
		self.previousLayer = previousLayer
		for index in range(0, len(self.neurons) - 1):
			self.neurons[index].setCountOfWeights(len(self.previousLayer.neurons))
	
	def transfer(self):
		for index in range(0, len(self.neurons) - 1):
			self.neurons[index].transfer()
		for index in range(0, len(self.neurons) - 1):
			self.neurons[index].activation()

	def error(self, targetActivationVector): #E_total = SUM of neurons-error (neuron.error)
		sum = 0
		activationVector = self.getActivationVector()
		for index in range(0, len(targetActivationVector)):
			sum += self.neurons[index].error(targetActivationVector[index])
		return sum

	def backpropagate(self, targetActivationVector):
		error = self.error(targetActivationVector)
		outputDelta = [0 for _ in range(0, len(targetActivationVector))]
		for index in range(0, len(self.neurons) - 1):
			self.neurons[index].backpropagate(error, targetActivationVector)
	
	def updateWeights(self):
		for neuron in self.neurons:
			neuron.updateWeights()
			
class HiddenLayer (InputLayer, OutputLayer):
	def __init__(self, countOfNeurons, net):
		super(HiddenLayer, self).__init__(countOfNeurons, net)
