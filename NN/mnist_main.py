import network
import numpy as np
import mnist_loader

training_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0,)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
