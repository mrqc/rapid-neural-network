import net
import layer
import random
import neuron

print "Initialising the training data (golden standard)"
netInputVector = [[0.05, 0.1], [0.1, 0.2]]
netOutputVector = [[0.01, 0.99], [0.04, 0.99]]

print "Setting up the net"
net = net.Net()
inputLayer = layer.InputLayer(2, net, neuron.SigmoidNeuron)
net.setInputLayer(inputLayer)
hiddenLayer1 = layer.HiddenLayer(3, net, neuron.SigmoidNeuron)
net.addHiddenLayer(hiddenLayer1)
hiddenLayer2 = layer.HiddenLayer(5, net, neuron.SigmoidNeuron)
net.addHiddenLayer(hiddenLayer2)
outputLayer = layer.OutputLayer(2, net, neuron.SigmoidNeuron)
net.setOutputLayer(outputLayer)

print "Training the net"
#for index in range(0, len(netInputVector)):
#	net.inputLayer.setActivationVector(netInputVector[index])
#	net.perform()
#	error = net.outputLayer.error(netOutputVector[index])
#	epoch = 1
#	while error > 0.01:
#		net.backpropagate(netOutputVector[index])
#		net.perform()
#		error = net.outputLayer.error(netOutputVector[index])
#		epoch += 1
#	print net
#	print "Training epochs: " + str(epoch)
#	print "Input: " + str(net.inputLayer.getActivationVector())
#	print "Golden-Standard: " + str(netOutputVector[index])
#	print "Output: " + str(net.outputLayer.getActivationVector())
#	print "Error after training: " + str(error)

error = 1
while error > 0.01:
	error = 0
	for index in range(0, len(netInputVector)):
		net.inputLayer.setActivationVector(netInputVector[index])
		net.backpropagate(netOutputVector[index])
	for index in range(0, len(netInputVector)):
		net.inputLayer.setActivationVector(netInputVector[index])
		net.perform()
		error += net.outputLayer.error(netOutputVector[index])
	print "Error after training: " + str(error)

print "Testing the net"
for index in range(0, len(netInputVector)):
	net.inputLayer.setActivationVector(netInputVector[index])
	net.perform()
	print "Error for test data " + str(index) + ": " + str(net.outputLayer.error(netOutputVector[index]))
