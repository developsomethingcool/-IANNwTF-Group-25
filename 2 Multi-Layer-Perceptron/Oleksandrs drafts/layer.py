import numpy as np

def ReLu(self, x):
    if x > 0:
        return x
    else:
        return 0

def dReLu(self, x):
    if x > 0:
        return 1
    else:
        return 0

class Layer():
    def __init__(self, n_units, input_units, input):
        self.n_units = n_units
        self.input_units = input_units
        self.weights = np.array([[np.random.randn() for i in range(0,n_units)] for i in range(0,input_units)])
        self.bias = np.array([np.random.randn() for i in range(0, n_units)])
        self.input = input
        self.preactivation = np.random.randn()
        self.activation = np.random.randn()
        self.gradient_input = np.random.randn()

    def forward_step(self, input):
        self.preactivation = self.weights @  input +  self.bias
        self.activation = ReLu(self.preactivation)
 
    def backward_step(self, learning_rate, error):
        gradient_weights = self.input.T * (dReLu(self.preactivation) @ error)
        gradient_bias = (dReLu(self.preactivation) @ error)
        self.gradient_input = (dReLu(self.preactivation) * error) * self.weights.T

        '''update the layer’s parameters'''
        self.weights = self.weights - learning_rate * gradient_weights
        self.bias = self.bias - learning_rate * gradient_bias
         




test_class = Layer(2,3, 3)
        

        