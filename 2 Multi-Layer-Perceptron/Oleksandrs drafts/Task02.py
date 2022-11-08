import numpy as np

def ReLu(self, x):
    if x > 0:
        return 1
    else:
        return 0

class Layer():
    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units
        self.weights = np.array([[np.random.randn() for i in range(0,n_units)] for i in range(0,input_units)])
        self.bias = np.array([np.random.randn() for i in range(0, n_units)])
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None

    def forward_step(self, x):
        o = self.weights @ x + self.bias
        if o > 0:
            return o
        else:
            return 0
 
    def backward_step(self, learning_rate):
        gradient_weights = self.layer_input.T * (ReLu(self.layer_preactivation) @ self.layer_activation)
        gradient_bias = self.layer_input.T * (ReLu(self.layer_preactivation) @ self.layer_activation)
        gradients_inputs = (ReLu(self.layer_preactivation) * self.layer_activation) * self.weights.T

        '''update the layer’s parameters'''
        self.weights = self.weights - learning_rate * gradients_inputs
        #self.bias = self.bias - learning_rate * gradients_inputs




test_class = Layer(2,3)
        

        