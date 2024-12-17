"""
WE're gonna simply apply MINE to learn mutual information between X and Y
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#define AGWN function as a tensorflow function
def AGWN(x, mu, sigma):
    # takes in a tensor x and returns a tensor y, where y is the additive gaussian white noise of x
    # x is a tensor of shape (n, 1)
    # mu is the mean of the gaussian noise
    # sigma is the standard deviation of the gaussian noise
    # y is a tensor of shape (n, 1)
    n = tf.shape(x)[0]
    x_type = x.dtype
    noise = tf.random.normal(shape=(n, 1), mean=mu, stddev=sigma, dtype=x_type)
    y = x + noise
    #now wrap around the values of y
    y = (y-x_min)%(x_max-x_min)+x_min
    return y


def BSC(x, p):
    # takes in a tensor x and returns a tensor y, y is x with a bit flip with probability p
    # x is a tensor of shape (n, 1)
    # p is the probability of flipping a value
    # y is a tensor of shape (n, 1)
    n = tf.shape(x)[0]
    x_type = x.dtype
    noise = tf.random.uniform(shape=(n, 1), minval=0, maxval=1, dtype=x_type)
    y = tf.where(noise<p, -x, x)
    return y

def dropout_Channel(x, p):
    # takes in a tensor x and returns a tensor y, y is x with a probability p of being zero
    # x is a tensor of shape (n, 1)
    # p is the probability of dropping out a value
    # y is a tensor of shape (n, 1)
    n = tf.shape(x)[0]
    x_type = x.dtype
    noise = tf.random.uniform(shape=(n, 1), minval=0, maxval=1, dtype=x_type)
    y = tf.where(noise<p, tf.zeros_like(x), x)
    return y

def cast_to_m_bit_precision(x,m, x_min, x_max):
    # takes in a tensor x and returns a tensor y, y is x with m-bit precision
    # x is a tensor of shape (n, 1)
    # m is the number of bits
    # x_min is the minimum value of x
    # x_max is the maximum value of x
    # y is a tensor of shape (n, 1)
    x_type = x.dtype
    x = tf.cast(x, tf.float32)
    x = (x-x_min)/(x_max-x_min)
    x = tf.cast(x*(2**m), tf.int32)
    x = tf.cast(x, tf.float32)
    x = x/(2**m)
    x = x*(x_max-x_min)+x_min
    x = tf.cast(x, x_type)
    return x


class NN(tf.keras.Model):
    def __init__(self, hidden_size=128, number_of_hidden_layers=3):
        super(NN, self).__init__()
        self.dense_layers = []
        for i in range(number_of_hidden_layers):
            self.dense_layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
        self.dense_layers.append(tf.keras.layers.Dense(1, activation='linear'))

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

@tf.function
def mutual_information_estimator(joint, marginal):
    # implements the mutual information estimator from MINE, using the Donsker-Varadhan theorem
    # we need to create marginal and joint distributions, then we can evaluate the estimator from MINE 
    return tf.reduce_mean(joint) - tf.math.log(tf.reduce_mean(tf.math.exp(marginal)))

# now we can train the model
# define the loss function
@tf.function
def loss(joint, marginal):
    #evaluate the model on the joint and marginal distributions
    return mutual_information_estimator(joint, marginal)

# define the optimizer with momentum
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.1)


# define the training loop without batches
def train(x, y, model, optimizer, epochs=10, validation_set=None):
    #history is a dictionary that stores the loss values and the validation loss values
    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        joint = tf.concat([x, y], axis=1)
        marginal = tf.concat([x, tf.random.shuffle(y)], axis=1)
        with tf.GradientTape() as tape:
            joint_model = model(joint)
            marginal_model = model(marginal)
            loss_value = -loss(joint_model, marginal_model)
        #update the largest loss value, print if a new max is set
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
        loss_value = -loss_value.numpy()
            
        if validation_set is not None:
            x_val, y_val = validation_set
            joint_val = tf.concat([x_val, y_val], axis=1)
            marginal_val = tf.concat([x_val, tf.random.shuffle(y_val)], axis=1)
            joint_val_model = model(joint_val)
            marginal_val_model = model(marginal_val)
            loss_val = loss(joint_val_model, marginal_val_model).numpy()
            history['val_loss'].append(loss_val)
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {}, Validation Loss: {}".format(epoch, loss_value, loss_val))
        else: #if there is no validation set, set the validation loss to -1
            history['val_loss'].append(-1)
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {}".format(epoch, loss_value))
        history['loss'].append(loss_value)
    return history, model

#define training loop using batches
def train_batch(x, y, model, optimizer, epochs=10, batch_size=100, validation_set=None):
    #history is a dictionary that stores the loss values and the validation loss values
    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        joint = tf.concat([x, y], axis=1)
        marginal = tf.concat([x, tf.random.shuffle(y)], axis=1)
        #create batches
        batches = np.arange(0, len(x), batch_size)
        #shuffle batches
        np.random.shuffle(batches)
        for batch in batches:
            joint_batch = joint[batch:batch+batch_size]
            marginal_batch = marginal[batch:batch+batch_size]
            with tf.GradientTape() as tape:
                joint_model = model(joint_batch)
                marginal_model = model(marginal_batch)
                loss_value = -loss(joint_model, marginal_model)
            #update the largest loss value, print if a new max is set
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
            loss_value = -loss_value.numpy()
        if validation_set is not None:
            x_val, y_val = validation_set
            joint_val = tf.concat([x_val, y_val], axis=1)
            marginal_val = tf.concat([x_val, tf.random.shuffle(y_val)], axis=1)
            joint_val_model = model(joint_val)
            marginal_val_model = model(marginal_val)
            loss_val = loss(joint_val_model, marginal_val_model).numpy()
            history['val_loss'].append(loss_val)
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {}, Validation Loss: {}".format(epoch, loss_value, loss_val))
        else: #if there is no validation set, set the validation loss to -1
            history['val_loss'].append(-1)
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {}".format(epoch, loss_value))
        history['loss'].append(loss_value)
    return history, model

def MINE(x, y,batch_train=True, hotload_model = None, model_hidden_size=128, model__numb_hidden_layers=3, learning_rate=0.001, epochs=10, batch_size=100, verbose=False, validation_set=None, optimiser = "Adam"):
    # Calculate the mutual information between x and y using the MINE algorithm
    # x is a tensor of shape (n, 1), y is a tensor of shape (n, 1), samples from the distrubution p(x, y)
    # model_hidden_size is the number of neurons in each hidden layer of the model
    # model__numb_hidden_layers is the number of neurons in the input layer of the model
    # learning_rate is the learning rate of the optimizer (adam)[]
    # epochs is the number of epochs to train the model
    # batch_size is the size of the batches used in the training loop

    #if a model is hotloaded, use that model
    if hotload_model is not None:
        model = hotload_model
    else:
        #define the model
        model = NN(model_hidden_size, model__numb_hidden_layers)
    #define the optimizer
    if optimiser == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimiser == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else: 
        print("No valid optimiser selected, using Adam")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #train the model
    if batch_train:
        history, model = train_batch(x, y, model, optimizer, epochs, batch_size, validation_set)
    else:
        history, model = train(x, y, model, optimizer, epochs, validation_set)

    #calculate the mutual information
    joint = tf.concat([x, y], axis=1)
    marginal = tf.concat([x, tf.random.shuffle(y)], axis=1)
    MI = mutual_information_estimator(model(joint), model(marginal)).numpy()
    if verbose:
        # plot the loss function
        plt.plot(history['loss'])
        plt.title("Loss function of MINE")
        plt.xlabel("Epoch")
        plt.ylabel("Estimation of the mutual information")
        plt.show()
    return MI, model, history




if __name__ == "__main__":
    # Here is a simple example of how to use the MINE algorithm to calculate the mutual information between two variables
    # We've defined a bunch of different distrubutions to try is out on.

    # Data_type:
    Data_type = 1 # 0 for AGWN, 1 for BSC, 2 for binary with dropout, 3 for m-bit precision

    if Data_type == 0: # AGWN
        #define data
        n = 10000
        mu = 0
        sigma = 0.01
        x_min = -10
        x_max = 10
        x = np.random.uniform(x_min, x_max, size=(n, 1))
        #check the maximum and minimum values of x
        print(np.max(x), np.min(x))
        y = AGWN(x, mu, sigma)
        var = sigma**2
        theoretical_MI = np.log(x_max-x_min)-0.5*np.log(2*np.pi*np.e*var)
        print("my calculations indicate the mutual information here is ", theoretical_MI)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        model = NN()
        epochs = 1000

    if Data_type == 1: # BSC
        n = 1000
        # x is array of integers, either -1 or 1
        x = tf.random.uniform(shape=(n, 1), minval=0, maxval=2, dtype=tf.int32)
        x = tf.cast(x, tf.float32)
        x = 2*x-1
        dropout = 0.2
        y = BSC(x, dropout)
        theoretical_MI = np.log(2)+dropout*np.log(dropout)+(1-dropout)*np.log(1-dropout)
        print("my calculations indicate the mutual information here is ", theoretical_MI)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model = NN(hidden_size=32, number_of_hidden_layers=2)
        epochs = 100

    if Data_type == 2: # binary with dropout
        n = 1000
        # x is array of integers, either -1 or 1
        x = tf.random.uniform(shape=(n, 1), minval=0, maxval=2, dtype=tf.int32)
        x = tf.cast(x, tf.float32)
        x = 2*x-1
        dropout = 0.5
        y = dropout_Channel(x, dropout)
        theoretical_MI = np.log(2)-dropout*np.log(2)
        print("my calculations indicate the mutual information here is ", theoretical_MI)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model = NN(hidden_size=32, number_of_hidden_layers=2)
        epochs = 100

    if Data_type == 3: # m-bit precision
        n = 10000
        # x is array of floats 
        x_min = -10
        x_max = 10
        x = np.random.uniform(x_min/100, x_max/100, size=(n, 1))*100
        m = 10
        y = cast_to_m_bit_precision(x, m, x_min, x_max)
        theoretical_MI = np.log(2**m)
        print("my calculations indicate the mutual information here is ", theoretical_MI)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model = NN()
        epochs = 100



    joint = tf.concat([x, y], axis=1)
    model(joint)

    # train the model 
    # use train_batch for batches
    history, _ = train_batch(x, y, model, optimizer, epochs=epochs, batch_size=128)
    # use train for no batches
    #history = train(x, y, model, optimizer, epochs=epochs)

    print(history)
    #evaluate the model
    joint = tf.concat([x, y], axis=1)
    marginal = tf.concat([x, tf.random.shuffle(y)], axis=1)
    MI = mutual_information_estimator(model(joint), model(marginal)).numpy()
    print("my experiments indicate the mutual information here is ", MI)

    # plot the loss function
    plt.plot(history)
    plt.plot([theoretical_MI]*len(history), label="theoretical MI")
    plt.legend()
    #set axis limits
    plt.xlim(0, len(history))
    plt.ylim(0, np.max([np.max(history), theoretical_MI+0.1]))
    plt.show()

