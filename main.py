import net
import layer
import random

net = net.Net()

inputLayer = layer.InputLayer(2, net)
net.setInputLayer(inputLayer)

hiddenLayer1 = layer.HiddenLayer(2, net)
net.addHiddenLayer(hiddenLayer1)

#hiddenLayer2 = layer.HiddenLayer(2, net)
#net.addHiddenLayer(hiddenLayer2)

outputLayer = layer.OutputLayer(2, net)
net.setOutputLayer(outputLayer)

netInputVector = [0.05, 0.1]
net.inputLayer.setActivationVector(netInputVector)

hiddenLayer1.neurons[0].weights = [0.15, 0.20, 0.35]
hiddenLayer1.neurons[1].weights = [0.25, 0.30, 0.35]
outputLayer.neurons[0].weights = [0.4, 0.45, 0.60]
outputLayer.neurons[1].weights = [0.5, 0.55, 0.60]

print "Printing the net - at this stage it has random weights"
print net

net.perform()

print "Printing all layer outputs after one run on the random weights"
print net.inputLayer.getActivationVector()
for hiddenLayer in net.hiddenLayers:
	print hiddenLayer.getActivationVector()
netOutputVectorLive = net.outputLayer.getActivationVector()
print netOutputVectorLive

net.training = True
netOutputVector = [random.choice([0, 1]) for _ in range(0, len(outputLayer.neurons))]
netOutputVector = [0.01, 0.99]
print "\nError Before Training"
print net.outputLayer.error(netOutputVector)
raw_input("press any key....")
print "\n\nTraining (backpropagation) the net on the netInputVector with netOutputVector as golden-standard"
for trainCycle in range(0, 1000):
	print "Train cycle " + str(trainCycle)
	net.backpropagate(netOutputVector)
print net
net.training = False
print "Error After Training"
net.perform()
print net.outputLayer.error(netOutputVector)
