import net
import layer

net = net.Net()

inputLayer = layer.InputLayer(4)
net.setInputLayer(inputLayer)

hiddenLayer1 = layer.HiddenLayer(8, inputLayer)
net.addHiddenLayer(hiddenLayer1)

hiddenLayer2 = layer.HiddenLayer(8, hiddenLayer1)
net.addHiddenLayer(hiddenLayer2)

outputLayer = layer.OutputLayer(8, hiddenLayer2)
net.setOutputLayer(outputLayer)

print net

