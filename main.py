import net
import layer
import random

net = net.Net()

inputLayer = layer.InputLayer(4)
net.inputLayer = inputLayer

hiddenLayer1 = layer.HiddenLayer(12)
inputLayer.connectNextLayer(hiddenLayer1)
hiddenLayer1.connectPreviousLayer(inputLayer)
net.addHiddenLayer(hiddenLayer1)

hiddenLayer2 = layer.HiddenLayer(8)
hiddenLayer1.connectNextLayer(hiddenLayer2)
hiddenLayer2.connectPreviousLayer(hiddenLayer1)
net.addHiddenLayer(hiddenLayer2)

outputLayer = layer.OutputLayer(16)
outputLayer.connectPreviousLayer(hiddenLayer2)
net.outputLayer = outputLayer

print "Printing the net - at this stage it has random weights"
print net

netInputVector = [random.choice([0, 1]) for _ in range(0, len(inputLayer.neurons))]
net.inputLayer.outputVector = netInputVector
net.perform()
print "Printing all layer outputs after one run on the random weights"
print net.inputLayer.outputVector
for hiddenLayer in net.hiddenLayers:
#	print [neuron.energy for neuron in hiddenLayer.neurons]
#	print [neuron.activationValue for neuron in hiddenLayer.neurons]
	print hiddenLayer.outputVector
#print [neuron.energy for neuron in outputLayer.neurons]
#print [neuron.activationValue for neuron in outputLayer.neurons]
netOutputVectorLive = net.outputLayer.outputVector
print netOutputVectorLive

print "Training (backpropagation) the net on the netInputVector with netOutputVector as golden-standard"
netOutputVector = [random.choice([0, 1]) for _ in range(0, len(outputLayer.neurons))]

