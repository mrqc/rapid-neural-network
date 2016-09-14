import net
import layer
import random

# net setup
net = net.Net()
inputLayer = layer.InputLayer(2, net)
net.setInputLayer(inputLayer)
hiddenLayer1 = layer.HiddenLayer(2, net)
net.addHiddenLayer(hiddenLayer1)
outputLayer = layer.OutputLayer(2, net)
net.setOutputLayer(outputLayer)

netInputVector = [0.05, 0.1]
net.inputLayer.setActivationVector(netInputVector)
hiddenLayer1.neurons[0].weights = [0.15, 0.20, 0.35]
hiddenLayer1.neurons[1].weights = [0.25, 0.30, 0.35]
outputLayer.neurons[0].weights = [0.4, 0.45, 0.60]
outputLayer.neurons[1].weights = [0.5, 0.55, 0.60]
print net
net.perform()
for hiddenLayer in net.hiddenLayers:
	print hiddenLayer.getActivationVector()
print net.outputLayer.getActivationVector()

net.training = True
netOutputVector = [0.01, 0.99]
print "Error of net:", net.outputLayer.error(netOutputVector)
raw_input("press any key....")

for trainCycle in range(0, 10000):
	print "Train cycle " + str(trainCycle)
	net.backpropagate(netOutputVector)
print net
net.training = False
net.perform()
print "Error: " + str(net.outputLayer.error(netOutputVector))
print "Input: " + str(net.inputLayer.getActivationVector())
print "Golden-Standard: " + str(netOutputVector)
print "Output: " + str(net.outputLayer.getActivationVector())
