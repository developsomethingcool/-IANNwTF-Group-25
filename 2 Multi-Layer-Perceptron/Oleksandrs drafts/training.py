import numpy as np
import mlp

def MSE(y, t):
    return 1/2 * (y-t) ** 2

training_mlp = mlp.MLP(10, 1, 1, 3, np.array([[1], [1]]))
epochs = 10

for i in range(epochs):
    res = training_mlp.forward_step(2)
    print("mlp output: ", res)

print( np.array([[1], [1]]).size)


