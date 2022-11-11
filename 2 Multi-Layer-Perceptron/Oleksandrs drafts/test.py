import numpy as np
import mlp

'''np_matrix = np.array(([1,2], [5,6],[8,9]))
print(np_matrix)

np_matrix2 = np.array([[np.random.randn() for i in range(1,3)] for i in range(1,4)])
print(np_matrix2)'''

print( np.array([[1], [1]]).size)

training_mlp = mlp.MLP(10, 1, 1, 3, np.array([[1], [1]]))
print("size of weights:", training_mlp.hidden_layer.weights.T.shape)
print(np.random.rand(1, 10).shape)
print( np.array([[1], [1]]).shape)