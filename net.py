import layer

# the depth of a NN is defined as sum of hidden layers and the output layer
# e.g. if the depth is 3, the net has an input layer, 2 hidden layers and one output layer
# so the definitive layers count is 4
class Net:
	def __init__(self):
		self.inputLayer = None
		self.hiddenLayers = []
		self.outputLayer = None
		self.training = False
	
	def addHiddenLayer(self, layer):
		self.hiddenLayers.append(layer)
	
	def __str__(self):
		hiddenLayersString = ""
		hiddenLayerCount = 0
		for hiddenLayer in self.hiddenLayers:
			hiddenLayerCount += 1
			hiddenLayersString += "Hidden Layer " + str(hiddenLayerCount) + ":\n" + str(hiddenLayer) + "\n"
		return "Input Layer:\n" + str(self.inputLayer) + "\n" + hiddenLayersString + "Output Layer:\n" + str(self.outputLayer)
	
	def getDefaultInputVector(self):
		return [0 for _ in range(0, len(self.inputLayer.weights))]
	
	def perform(self):
		for hiddenLayer in self.hiddenLayers:
			hiddenLayer.transfer()
		self.outputLayer.transfer()
	
	def backpropagate(self, targetOutputVector):
		self.outputLayer.backpropagate(targetOutputVector)
		for hiddenLayer in reversed(self.hiddenLayers):
			hiddenLayer.backpropagate(hiddenLayer.getActivationVector())
		self.outputLayer.updateWeights()
		for hiddenLayer in reversed(self.hiddenLayers):
			hiddenLayer.updateWeights()
		
