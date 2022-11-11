import Task02

class MLP():
    def __init__(self, n_hidden_layers, n_units):
        self.n_hidden_layers = n_hidden_layers
        self.n_units = n_units
        self.layer = Task02.Layer(n_units, ) #[for i in range(n_hidden_layers)]

    def forward_step(self):
        pass

    def baclpropagantion(self):
        pass