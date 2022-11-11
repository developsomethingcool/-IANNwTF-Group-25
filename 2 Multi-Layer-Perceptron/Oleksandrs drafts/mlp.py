import layer
import numpy as np

class MLP():
    def __init__(self, n_hidden_units, n_input_units, n_output_units, n_layers, input):
        self.n_hidden_layers = n_hidden_units
        self.n_input_units = n_input_units
        self.n_output_units = n_output_units
        self.n_layers = n_layers
        self.input = input
        self.hidden_layer = layer.Layer(n_hidden_units, n_input_units, input) #[for i in range(n_hidden_layers)]
        self.output_layer = layer.Layer(n_output_units, n_hidden_units, input=np.empty([n_hidden_units,1]))

    def forward_step(self, n_layers):
        #for i in range(n_layers-1):
        self.input = self.hidden_layer.forward_step(self.input)
        input = self.output_layer.forward_step(self.input)
        return input

    def backpropagantion(self, learning_rate, error):
        self.output_layer.backward_step(learning_rate, error)
        self.output_layer.backward_step(learning_rate, self.output_layer.gradient_input)

MLP(10,1, 1, 1, [1,1])

