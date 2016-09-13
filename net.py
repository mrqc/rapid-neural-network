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
		self.learningRate = 0.5
		self.learningRateBias = 0.05
	
	def addHiddenLayer(self, layer):
		self.hiddenLayers.append(layer)
		if len(self.hiddenLayers) == 1:
			self.inputLayer.connectNextLayer(self.hiddenLayers[0])
			self.hiddenLayers[0].connectPreviousLayer(self.inputLayer)
		else:
			self.hiddenLayers[len(self.hiddenLayers) - 2].connectNextLayer(self.hiddenLayers[len(self.hiddenLayers) - 1])
			self.hiddenLayers[len(self.hiddenLayers) - 1].connectPreviousLayer(self.hiddenLayers[len(self.hiddenLayers) - 2])
	
	def setInputLayer(self, inputLayer):
		self.inputLayer = inputLayer
	
	def setOutputLayer(self, outputLayer):
		self.outputLayer = outputLayer
		self.hiddenLayers[len(self.hiddenLayers) - 1].connectNextLayer(self.outputLayer)
		self.outputLayer.connectPreviousLayer(self.hiddenLayers[len(self.hiddenLayers) - 1])
	
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
			hiddenLayer.backpropagate(targetOutputVector)
		self.outputLayer.updateWeights()
		for hiddenLayer in reversed(self.hiddenLayers):
			hiddenLayer.updateWeights()
