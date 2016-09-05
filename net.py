import layer

# the depth of a NN is defined as sum of hidden layers and the output layer
# e.g. if the depth is 3, the net has an input layer, 2 hidden layers and one output layer
# so the definitive layers count is 4
class Net:
	def __init__(self):
		self.inputLayer = None
		self.hiddenLayers = []
		self.outputLayer = None
	
	def addHiddenLayer(self, layer):
		self.hiddenLayers.append(layer)
	
	def setInputLayer(self, layer):
		self.inputLayer = layer
	
	def setOutputLayer(self, layer):
		self.outputLayer = layer
	
	def __str__(self):
		hiddenLayersString = ""
		hiddenLayerCount = 0
		for hiddenLayer in self.hiddenLayers:
			hiddenLayerCount += 1
			hiddenLayersString += "Hidden Layer " + str(hiddenLayerCount) + ":\n" + str(hiddenLayer) + "\n" * 2
		return "Input Layer:\n" + str(self.inputLayer) + "\n" * 2 + hiddenLayersString + "Output Layer:\n" + str(self.outputLayer)
