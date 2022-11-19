import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

'''Loading the MNIST dataset'''
(train_ds, test_ds) = tfds.load('mnist', split =[ 'train', 'test'], as_supervised =True , with_info = False)

#print(ds_info)
#tfds.show_examples(train_ds, ds_info)

'''Setting up the data pipeline'''
def prepare_mnist_data(mnist):
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    mnist = mnist.cache()
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(16)
    mnist = mnist.prefetch(20)
    return mnist

'''Building a deep neural network with TensorFlow Class'''

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x

def train_step(model, input, target, loss_function, optimizer):
      # loss_object and optimizer_object are instances of respective tensorflow classes
      with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
      return loss

def test(model, test_data, loss_function):
        '''returns complete average loss and complete average accuracy overall testing data'''

        test_accuracy_aggregator = []
        test_loss_aggregator = []

        for (input, target) in test_data:
            prediction = model(input)
            sample_test_loss = loss_function(target, prediction)
            sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            sample_test_accuracy = np.mean(sample_test_accuracy)
            
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

        return test_loss, test_accuracy
    
'''Visualization'''
def visualization( train_losses , test_losses , test_accuracies):
  plt.figure()
  line1, = plt.plot(train_losses)
  line2, = plt.plot(test_losses)
  line3, = plt.plot(test_accuracies)
  plt.xlabel("Training steps")
  plt.ylabel("Loss/Accuracy")
  plt.legend((line1,line2, line3),("training","test", "test accuracy"))
  plt.show()


train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)

"smaller set"
train_dataset = train_dataset.take(1000)
test_dataset = test_dataset.take(100)

print("Size train set:", len(train_dataset))
print("Size test set:", len(test_dataset))

num_epochs = 10
learning_rate = 0.001

model = MyModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

train_losses = []
test_losses = []
test_accuracies = []

# test model before training
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training 
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input,target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # track accuracy and test loss
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

print("Plotting result")
visualization ( train_losses , test_losses , test_accuracies )


