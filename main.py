import net
import layer

net = net.Net()

inputLayer = layer.InputLayer(4)
net.inputLayer = inputLayer

hiddenLayer1 = layer.HiddenLayer(8)
inputLayer.connectNextLayer(hiddenLayer1)
hiddenLayer1.connectPreviousLayer(inputLayer)
net.addHiddenLayer(hiddenLayer1)

hiddenLayer2 = layer.HiddenLayer(8)
hiddenLayer1.connectNextLayer(hiddenLayer2)
hiddenLayer2.connectPreviousLayer(hiddenLayer1)
net.addHiddenLayer(hiddenLayer2)

outputLayer = layer.OutputLayer(8)
outputLayer.connectPreviousLayer(hiddenLayer2)
net.outputLayer = outputLayer

print net

netInputVector = [1, 0, 0, 1]
net.inputLayer.outputVector = netInputVector
net.perform()
print net.inputLayer.outputVector
for hiddenLayer in net.hiddenLayers:
	print [neuron.energy for neuron in hiddenLayer.neurons]
	print [neuron.activationValue for neuron in hiddenLayer.neurons]
	print hiddenLayer.outputVector
print [neuron.energy for neuron in outputLayer.neurons]
print [neuron.activationValue for neuron in outputLayer.neurons]
print net.outputLayer.outputVector

