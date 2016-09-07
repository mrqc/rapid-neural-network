import net
import layer

net = net.Net()

inputLayer = layer.InputLayer(10000)
net.inputLayer = inputLayer

hiddenLayer1 = layer.HiddenLayer(10000)
inputLayer.connectNextLayer(hiddenLayer1)
hiddenLayer1.connectPreviousLayer(inputLayer)
net.addHiddenLayer(hiddenLayer1)

hiddenLayer2 = layer.HiddenLayer(1000)
hiddenLayer1.connectNextLayer(hiddenLayer2)
hiddenLayer2.connectPreviousLayer(hiddenLayer1)
net.addHiddenLayer(hiddenLayer2)

outputLayer = layer.OutputLayer(10000)
outputLayer.connectPreviousLayer(hiddenLayer2)
net.outputLayer = outputLayer

#print net

netInputVector = [1 for _ in range(0, 10000)]
net.inputLayer.outputVector = netInputVector
net.perform()
#print net.inputLayer.outputVector
#for hiddenLayer in net.hiddenLayers:
#	print [neuron.energy for neuron in hiddenLayer.neurons]
#	print [neuron.activationValue for neuron in hiddenLayer.neurons]
#	print hiddenLayer.outputVector
#print [neuron.energy for neuron in outputLayer.neurons]
#print [neuron.activationValue for neuron in outputLayer.neurons]
#print net.outputLayer.outputVector

