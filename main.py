import net
import layer
import random

print "Initialising the training data (golden standard)"
netInputVector = [[0.05, 0.1], [0.1, 0.9]]
netOutputVector = [[0.01, 0.99], [0.5, 0.5]]

print "Setting up the net"
net = net.Net()
inputLayer = layer.InputLayer(2, net)
net.setInputLayer(inputLayer)
hiddenLayer1 = layer.HiddenLayer(3, net)
net.addHiddenLayer(hiddenLayer1)
hiddenLayer2 = layer.HiddenLayer(4, net)
net.addHiddenLayer(hiddenLayer2)
outputLayer = layer.OutputLayer(2, net)
net.setOutputLayer(outputLayer)

print "Training the net"
for index in range(0, len(netInputVector)):
	net.inputLayer.setActivationVector(netInputVector[index])
	net.perform()
	error = net.outputLayer.error(netOutputVector[index])
	epoch = 1
	while error > 0.01:
		net.backpropagate(netOutputVector[index])
		net.perform()
		error = net.outputLayer.error(netOutputVector[index])
		epoch += 1
	#print net
	print "Training epochs: " + str(epoch)
	print "Input: " + str(net.inputLayer.getActivationVector())
	print "Golden-Standard: " + str(netOutputVector[index])
	print "Output: " + str(net.outputLayer.getActivationVector())
	print "Error after training: " + str(error)

print "Testing the net"
for index in range(0, len(netInputVector)):
	net.inputLayer.setActivationVector(netInputVector[index])
	net.perform()
	print "Error for test data " + str(index) + ": " + str(net.outputLayer.error(netOutputVector[index]))
