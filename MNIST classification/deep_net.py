import tensorflow_datasets as tfds
import tensorflow as tf

'''2.1. Loading the MNIST dataset'''
(train_ds, test_ds), ds_info = tfds.load('mnist', split =[ 'train', 'test'], as_supervised =True , with_info = True)

print(ds_info)
#tfds.show_examples(train_ds, ds_info)

'''2.2. Setting up the data pipeline'''

'''2.3. Building a deep neural network with TensorFlow'''

'''2.4 Training the network'''

'''2.5 Visualization'''

'''3. Adjusting the hyperparameters of your model'''

