"""
We'll just quickly train a very simple NN on a basic task.
We'll use the MINE algorithm to estimate the mutual information between the input and the output of some of it's layers
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from MINE import MINE, mutual_information_estimator

### program settings
# which model settings
train_new_model = False
import_model = True
model_path = "model3_rand.h5"  #"model3.h5" for the trained non random model
save_model = False
check_input_output = False

#MI settings
MI_input_output = False
MI_input_layer = False
MI_layer_to_layer = False
inspect_MI_graphs = False
#Stuff we're currently using
MI_input_layer_gen_data = False
MI_input_layer_plot = True

presets = 0
if presets != 0:
    preset = [[True, False, True, False, True, False, False, False, False, False],# 1, train new model
              [False, True, False, False, False, True, False, False, False, False],# 2, input to layer k
              [False, False, False, False, False, False, False, False, False, False],# 3, false layer, does nothing
              [False, True, False, False, False, False, False, False, True, False],# 4, input to layer k, gen data
              [False, True, False, False, False, False, False, False, False, True]]# 5, input to layer k, plot data
    train_new_model, import_model, save_model, check_input_output, MI_input_output, MI_input_layer, MI_layer_to_layer, inspect_MI_graphs, MI_input_layer_gen_data, MI_input_layer_plot = preset[presets-1]

bits_of_information= 3 #number of bits (no longer bits) of information being passed through the network

#set dpi in matplot lib
plt.rcParams['figure.dpi'] = 200


### train the NN on some simple data

# Define a simple NN
def NN(model_hidden_size, model__numb_hidden_layers, input_shape=(1,), output_shape=(1,)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(model_hidden_size, input_shape=input_shape, activation='relu'))
    for i in range(model__numb_hidden_layers):
        model.add(tf.keras.layers.Dense(model_hidden_size, activation='relu'))
    model.add(tf.keras.layers.Dense(output_shape[0], activation='linear'))
    return model

def NN_dropout(model_hidden_size, model__numb_hidden_layers, input_shape=(1,), output_shape=(1,), dropout=0.1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(model_hidden_size, input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    for i in range(model__numb_hidden_layers):
        model.add(tf.keras.layers.Dense(model_hidden_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_shape[0], activation='linear'))
    return model

def NN_sparse_dropout(model_hidden_size, model__numb_hidden_layers,dropout_every_k=5, input_shape=(1,), output_shape=(1,), dropout=0.1):
    model = tf.keras.Sequential()
    inverse_of_dropout_scaling = (1-dropout)
    for i in range(model__numb_hidden_layers):
        model.add(tf.keras.layers.Dense(model_hidden_size, activation='tanh'))
        if i % dropout_every_k == dropout_every_k-1:
            model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Lambda(lambda x: x * inverse_of_dropout_scaling))
    model.add(tf.keras.layers.Dense(output_shape[0], activation='linear'))
    return model



# Define the some data
# x is a 10 dim vector
# y is a 2 dim
def data(n=1000):
    # x uniform distrubution between -1 and 1
    x = np.random.uniform(-1, 1, (n, 10))
    # y_0 is 1 dim vector, sum of x modulo 2
    binary_x = np.where(x > 0, 1, 0) # useful for calculating y
    y_0 = np.sum(binary_x, axis=1) % 2
    #y[1] is 1 if x[0] > 0 else -1
    y_1 = np.where(x[:, 0] > 0, 1, -1)
    y = np.stack((y_0, y_1), axis=1)
    other_binary_x = np.where(x > 0, 1, -1)
    return other_binary_x, y


def data(n=1000): # random vector in the reals, y is subset of x
    x = np.random.uniform(-1, 1, (n, 10))
    # y is a copy of x[0]
    y = x[:, 0].reshape(-1, 1)
    return x, y

# Define the some data
# x is a 10 dim vector of random numbers between -1 and 1
# y is a 1 dim vector which is 1 if x[0] > 0 else -1
def data(n=1000): #random vector in the reals, y is "is x[0]>0?"
    x = np.random.uniform(-1, 1, (n, 10))
    #binary_x = np.where(x > 0, 1, -1)
    y = np.where(x[:, 0] > 0, 1, -1)
    y = y.reshape(-1, 1)
    return x, y

def data(n=1000, keep_first_i = 1):
    # generates data, x is a 10 dim vector of random numbers between -1 and 1, 
    # y is an keep_first_i dim vector, y[i]=1 if x[i] > 0 else -1
    x = np.random.uniform(-1, 1, (n, 10))
    y = np.where(x[:, 0:keep_first_i] > 0, 1, -1)
    y = y.reshape(-1, keep_first_i)
    return x, y

def data(n=1000, keep_first_i = 1):
    # generates data, x is a 10 dim vector of uniform random numbers between -1 and 1, 
    # y is an keep_first_i dim vector with y[i]=x[i]
    x = np.random.uniform(-1, 1, (n, 10))
    y = x[:, 0:keep_first_i]
    y = y.reshape(-1, keep_first_i)
    return x, y

def data(n=1000, keep_first_i = 1):
    dim_x = keep_first_i
    # generates data, x is a dim_x dim vector of uniform random numbers between -1 and 1, 
    # y is an x_dim dim vector with y[i]=x[i]
    x = np.random.uniform(-1, 1, (n, keep_first_i))
    y = x[:, 0:dim_x]
    y = y.reshape(-1, dim_x)
    return x, y

#set and print seed
seed = 44
tf.random.set_seed(seed)
np.random.seed(seed)
print("seed: {}".format(seed))

def new_model(random=False):
    # import simple MSE loss
    loss = tf.keras.losses.MeanSquaredError()

    #get data, model and train
    x,y = data(1000000, keep_first_i=bits_of_information)
    #hyperparameters
    model_hidden_size, model__numb_hidden_layers = 32, 40
    # old input size used to be fixed input_shape, output_shape = (10,), (bits_of_information,)
    input_shape, output_shape = (bits_of_information,), (bits_of_information,)
    #model = NN_dropout(model_hidden_size, model__numb_hidden_layers , input_shape, output_shape, dropout=0.1)
    model = NN_sparse_dropout(model_hidden_size, model__numb_hidden_layers, dropout_every_k=4, input_shape = input_shape, output_shape = output_shape, dropout=0.2)
    model(x[0:1])
    model.compile(loss=loss, optimizer='adam')

    if not random:
        # train with keras.fit
        history = model.fit(x, y, epochs=4, batch_size=100, verbose=1, validation_split=0.02) 
    
    return model

if train_new_model:
    model = new_model(random=True)

if import_model:
    model = tf.keras.models.load_model(model_path)

if save_model:
    print("saving model to {}".format(model_path))
    model.save(model_path)

if check_input_output:
    #for each i in range(bits_of_information), plot x[i] vs y[i]
    if bits_of_information == None:
        bits_of_information = 3
    x,y = data(100, keep_first_i=bits_of_information)
    y_pred = model(x, training=True)
    for i in range(bits_of_information):
        plt.scatter(x[:,i], y_pred[:,i], label="y_pred")
        plt.legend()
        plt.show()

def var(x):
    x = np.array(x)
    return np.mean(x**2) - np.mean(x)**2

#######################################
### Estimate the mutual information ###
#######################################

if MI_input_output: #estimate the mutual information between the input and the output of the NN
    # grab input distrubution:
    x,y = data(2000, keep_first_i=bits_of_information) #data(2000)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = model(x, training=True)
    #now let MINE estimate the mutual information
    MI, _, history = MINE(x, y, model_hidden_size=64, model__numb_hidden_layers=3, learning_rate=0.001, epochs=100, batch_size=100, verbose = 0)
    print("MI: {}, if this deviates too far from 0.69 we have a problem, too high indicates the output is still correlated with more than one input variable".format(MI))

def generate_distrubution_from_layer(model, x, l):
    # truncate the model at layer l and generate a distrubution from it
    # create surrogate model, which is the same as the original model, but truncated at layer l
    activation_model = tf.keras.Model(inputs=model.input,
                                    outputs=model.layers[l].output)
    activations = activation_model(x, training = True)
    return activations

def generate_distribution_from_layer_to_layer(model, x, l1, l2):
    # create a new model that outputs the activations of both l1 and l2
    raise notImplementedError("Dropout is currently off in this model, I will have to turn it on")
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=[model.layers[l1].output, model.layers[l2].output])
    # take x through the intermediate model
    activations_l1, activations_l2 = intermediate_layer_model.predict(x)
    return activations_l1, activations_l2


def MI_of_layer_l(model, l, verbose = False, k=0): # k is secret parameter that if more than 0 will generate a distrubution from layer l to layer l+k
    #now let's estimate the mutual information between the input and the output of some of the layers of the NN
    # generate distrubutions
    if k == 0:
        x,_ = data(40960, keep_first_i=bits_of_information)
        y = generate_distrubution_from_layer(model, x, l)
        x_val,_ = data(4096, keep_first_i=bits_of_information)
        y_val = generate_distrubution_from_layer(model, x_val, l)
    else:
        raise notImplementedError("This branch is now defunct")
        x,_ = data(40960)
        x, y = generate_distribution_from_layer_to_layer(model, x, l, l+k)
        x_val,_ = data(4096)
        x_val, y_val = generate_distribution_from_layer_to_layer(model, x_val, l, l+k)
    #now let MINE estimate the mutual information
    MI, MINE_model, history1 = MINE(x, y, model_hidden_size=64, model__numb_hidden_layers=3, learning_rate=0.0001, epochs=100, batch_size=1024, verbose = 0, validation_set=(x_val, y_val), optimiser='Adam')
    print("MI between input and layer {}: {}".format(l, MI))
    #let's push it a little higher to see if we can get a better estimate
    MI, MINE_model, history2 = MINE(x, y, hotload_model=MINE_model, model_hidden_size=64, model__numb_hidden_layers=3, learning_rate=0.000005, epochs=50, batch_size=2048, verbose = 0, validation_set=(x_val, y_val), optimiser='Adam')
    print("MI between input and layer {}: {}".format(l, MI))
    
    #append the two histories, history is a dict with the keys 'loss' and val_loss
    history = {}
    history['loss'] = history1['loss'] + history2['loss']
    history['val_loss'] = history1['val_loss'] + history2['val_loss']

    #calculate a rolling average of the loss+val_loss
    # set window size
    window_size = 10
    # calculate rolling average in new dict
    history_rolling = {}
    history_rolling['loss'] = np.convolve(history['loss'], np.ones((window_size,))/window_size, mode='valid')
    history_rolling['val_loss'] = np.convolve(history['val_loss'], np.ones((window_size,))/window_size, mode='valid')
    #calculating a rolling variance
    history_rolling['loss_var'] = np.convolve(history['loss'], np.ones((window_size,))/window_size, mode='valid')**2
    history_rolling['val_loss_var'] = np.convolve(history['val_loss'], np.ones((window_size,))/window_size, mode='valid')**2
    # add on the largest validation loss as "MI"
    history_rolling['MI'] = np.max(history_rolling['val_loss'])

    if verbose:
        #plot the history val loss
        plt.plot(history_rolling['val_loss'])
        # plot the variance as a shaded area
        plt.fill_between(np.arange(len(history_rolling['val_loss'])), history_rolling['val_loss'] - np.sqrt(history_rolling['val_loss_var']), history_rolling['val_loss'] + np.sqrt(history_rolling['val_loss_var']), alpha=0.5)
        
        plt.show()
    return history_rolling

if MI_input_layer:
    #check the layer names of each layer of model:
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    exit()
    rolling_histories=[]
    #simply goes through all layers and estimates the mutual information between the input and the output of the layer
    run_fresh = False
    # print(len(model.layers)) #There are 16 layers in the model
    ls = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    #ls = [1,3,5,7,9,11,13,15]
    if run_fresh:
        for l in ls:
            # fancy text displaying the layer number
            print(" "*100, end="\n")
            print("working on layer {}".format(l))
            print(" "*100, end="\n")
            history_rolling = MI_of_layer_l(model, l, verbose = 0)
            #save the history incase crash
            np.save("history_rolling_layer_{}_DD.npy".format(l), history_rolling)
            rolling_histories.append(history_rolling)
    else:
        #load the saved histories
        rolling_histories = []
        histories_to_load = ls #np.arange(0, len(model.layers), 5)
        for l in histories_to_load:
            rolling_histories.append(np.load("history_rolling_layer_{}_DD.npy".format(l), allow_pickle=True).item())
        ls = histories_to_load
    #print([rolling_histories1[l]['val_loss'][-1:][0] for l in range(len(rolling_histories))])
    
    #save rolling histories for multiple trials
    #YOU HAVE FILLED _1 and are filling _2 
    np.save("rolling_histories_D06_trail_2.npy", np.array(rolling_histories))
    #load rolling histories for multiple trials, we will plot the average of the rolling histories
    rolling_histories1 = np.load("rolling_histories_DD_1.npy", allow_pickle=True)
    rolling_histories2 = np.load("rolling_histories_DD_2.npy", allow_pickle=True)
    #rolling_histories3 = np.load("rolling_histories_D06_trail_3.npy", allow_pickle=True)

    # and extract what we want for the graph 
    final_val_loss1 = [rolling_histories1[l]['val_loss'][-1:][0] for l in range(len(rolling_histories1))]
    final_val_loss2 = [rolling_histories2[l]['val_loss'][-1:][0] for l in range(len(rolling_histories2))]
    plt.plot([0, len(model.layers)], [np.log(2)*16*0.7, np.log(2)*16*0.7], color='gold', label='theoretical limit of information transfer')

    plt.plot(ls, list(final_val_loss1), label='information content of each layer about the input')
    plt.plot(ls, list(final_val_loss2), label='information content of each layer about the input')
    plt.legend()
    plt.title("Mutual Information between input and each layer")
    plt.xlabel("Layer number")
    plt.ylabel("Mutual Information")
    plt.show()

def MI_data_generator(model, ls, trials_to_run, verbose = 0):
    for trial in trials_to_run:
        if verbose:
            print("working on trial {}".format(trial))
        rolling_histories=[]
        for l in ls:
            if verbose:
                print("working on layer {}".format(l))
            history_rolling = MI_of_layer_l(model, l, verbose = 0)
            rolling_histories.append(history_rolling)
        # used to be: np.save("history_rolling_layer_{}_DD_{}.npy".format(l, trial), history_rolling)
        np.save("history_rolling_layer_{}_DD_{}.npy".format(l, trial), rolling_histories)
if MI_input_layer_gen_data: # function that does the data collection part of the MI_input_layer clause but looped so we can collect data for multiple trials in one run.
    ls = [6,12,18,24,30,36,42,48,54,60]
    ls = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60]

    trails_to_run = ["RH_trail_1_D02_rand.npy", "RH_trail_2_D02_rand.npy", "RH_trail_3_D02_rand.npy"]
    MI_data_generator(model, ls, trails_to_run, verbose = 1)
    model = tf.keras.models.load_model("model3.h5")
    trails_to_run = ["RH_trail_1_D02.npy", "RH_trail_2_D02.npy", "RH_trail_3_D02.npy"]
    MI_data_generator(model, ls, trails_to_run, verbose = 1)

    exit() 
    for trail in trails_to_run:
        print("running trail {}".format(trail))
        rolling_histories=[]
        for l in ls:
            # fancy text displaying the layer number
            print(" "*100, end="\n")
            print("working on layer {}".format(l))
            print(" "*100, end="\n")
            history_rolling = MI_of_layer_l(model, l, verbose = 0)
            #save the history incase crash
            np.save("history_rolling_layer_{}_DD.npy".format(l), history_rolling)
            rolling_histories.append(history_rolling)
        #save rolling histories for multiple trials
        np.save(trail, np.array(rolling_histories))
    

if MI_input_layer_plot: # function that does the plotting part of the MI_input_layer clause but looped so we can plot data for multiple trials in one run.
    ls = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60]
    
    # Theoretical limits
    #theoretical limit for a dropout layer
    theory_limit_dropout = np.log(2)*32*0.7
    #theoretical limit for dense layer
    theory_limit_dense = np.log(2)*32
    #axe
    tl = [theory_limit_dense,theory_limit_dense, theory_limit_dropout, theory_limit_dropout]
    l_xaxis = [ls[0],2,2,ls[-1]]
    tl = [theory_limit_dropout, theory_limit_dropout]
    l_xaxis = [ls[0], ls[-1]]
    plt.plot(l_xaxis, tl, color='gold', label='theoretical limit')
    #minimum necessary information
    min_info = [np.log(2)*3, np.log(2)*3]
    l_xaxis = [ls[0], ls[-1]]
    plt.plot(l_xaxis, min_info, color='pink', label='necessary information')

    plot_random = True
    if plot_random:
        trails_to_plot = ["RH_trail_1_D02_rand.npy", "RH_trail_2_D02_rand.npy", "RH_trail_3_D02_rand.npy"]
        rolling_histories = []
        for trail in trails_to_plot:
            rolling_histories.append(np.load(trail, allow_pickle=True))
        # Calculate the mean and std of the rolling histories
        final_val_loss = [[rolling_histories1[l]['val_loss'][-1:][0] for l in range(len(rolling_histories1))] for rolling_histories1 in rolling_histories]
        final_val_loss = np.mean(final_val_loss, axis=0)
        final_val_loss_std = [[rolling_histories1[l]['val_loss'][-1:][0] for l in range(len(rolling_histories1))] for rolling_histories1 in rolling_histories]
        final_val_loss_std = np.std(final_val_loss_std, axis=0)
        #if any of the value are nan, replace with 100
        final_val_loss_std = [0 if np.isnan(x) else x for x in final_val_loss_std]
        final_val_loss = [100 if np.isnan(x) else x for x in final_val_loss]
        #print them out
        print("final_val_loss = ", final_val_loss)
        print("final_val_loss_std = ", final_val_loss_std)

        # Plot the mean and std of the rolling histories
        plt.plot(ls, list(final_val_loss), label='Random model')
        plt.fill_between(ls, np.array(final_val_loss) - np.array(final_val_loss_std), np.array(final_val_loss) + np.array(final_val_loss_std), alpha=0.5)


    # plot another set of trails to compare against:
    plot_learnt = True
    if plot_learnt:
        trails_to_plot = ["RH_trail_1_D02.npy", "RH_trail_2_D02.npy", "RH_trail_3_D02.npy"]
        rolling_histories = []
        for trail in trails_to_plot:
            rolling_histories.append(np.load(trail, allow_pickle=True))
        # Calculate the mean and std of the rolling histories
        final_val_loss = [[rolling_histories1[l]['val_loss'][-1:][0] for l in range(len(rolling_histories1))] for rolling_histories1 in rolling_histories]
        final_val_loss = np.mean(final_val_loss, axis=0)
        final_val_loss_std = [[rolling_histories1[l]['val_loss'][-1:][0] for l in range(len(rolling_histories1))] for rolling_histories1 in rolling_histories]
        final_val_loss_std = np.std(final_val_loss_std, axis=0)
        #if any of the value are nan, replace with 100
        final_val_loss_std = [0 if np.isnan(x) else x for x in final_val_loss_std]
        final_val_loss = [100 if np.isnan(x) else x for x in final_val_loss]

        # Plot the mean and std of the rolling histories
        plt.plot(ls, list(final_val_loss), label='Learnt model')
        plt.fill_between(ls, np.array(final_val_loss) - np.array(final_val_loss_std), np.array(final_val_loss) + np.array(final_val_loss_std), alpha=0.5)

    plt.title("Mutual Information between input and each layer of a DNN")
    plt.xlabel("Layer number")
    plt.ylabel("Mutual Information")



    #fix axis
    plt.legend()
    plt.ylim(0, 1.1*np.max(final_val_loss))
    plt.xlim(ls[0], ls[-1])
    plt.show()

        

if inspect_MI_graphs:
    # plot the final val loss for each l using the precalculated rolling histories for a given layer
    l = 15
    rolling_histories = np.load("history_rolling_layer_{}.npy".format(l), allow_pickle=True).item()
    plt.plot(rolling_histories['val_loss'])
    # plot the variance as a shaded area
    #set dpi
    plt.rcParams['figure.dpi'] = 300
    plt.fill_between(np.arange(len(rolling_histories['val_loss'])), rolling_histories['val_loss'] - np.sqrt(rolling_histories['val_loss_var']), rolling_histories['val_loss'] + np.sqrt(rolling_histories['val_loss_var']), alpha=0.5)
    plt.show()

if MI_layer_to_layer:
    #estimates the mutual information between the input and the output of the layer l and the input and the output of the layer l+k
    #this is done by first generating a distribution from the input to the layer l and then from the layer l to the layer l+k
    #the mutual information is then estimated by MINE
    rolling_histories=[]
    ls = [0, 3, 6, 9, 12, 15, 17]
    k = 3
    #run on every 4th layer to save time
    for l in ls: 
        # fancy text displaying the layer number
        print(" "*100, end="\n")
        print("Estimating MI between layer {} and layer {}".format(l, l+k))
        print(" "*100, end="\n")
        history_rolling = MI_of_layer_l(model, l, k=k, verbose = 0)
        #save the history incase crash
        np.save("history_rolling_layer_{}_to_{}_DD.npy".format(l, l+k), history_rolling)
        rolling_histories.append(history_rolling)
    
    # plot the final val loss for each l using the precalculated rolling histories
    plt.plot(ls, [rolling_histories[l]['val_loss'][-1:] for l in range(len(rolling_histories))])
    # give error bars as the variance of the rolling average
    # following is deprecated
    # plt.errorbar(np.arange(len(rolling_histories)), [rolling_histories[l]['val_loss'][-1:] for l in range(len(rolling_histories))], yerr=[np.sqrt(rolling_histories[l]['val_loss_var'][-1:]) for l in range(len(rolling_histories))], fmt='o')
    # plt.errorbar(np.arange(len(rolling_histories)), [rolling_histories[l]['val_loss'][-1:] for l in range(len(rolling_histories))], yerr=[np.sqrt(rolling_histories[l]['val_loss_var'][-1:]) for l in range(len(rolling_histories))], fmt='o', capsize=5)
    plt.show()

exit()


print_l_training_graph = 0
#plot the training graph for the layer print_l_training_graph
#plt.plot(rolling_histories[print_l_training_graph]['val_loss'])
#plt.title("Validation loss for layer {}".format(print_l_training_graph))
#plt.show()


x_axis = ls # list(np.arange(0, len(model.layers), 5))
y_axis = list([rolling_histories[l]['val_loss'][-1:][0] for l in range(len(rolling_histories))])
print(x_axis)
print(y_axis)
# plot the final val loss for each l using the precalculated rolling histories
plt.plot(x_axis, y_axis)
#plot minimum possible entropy (log(2)) as a straight line in gold
plt.plot([0, len(model.layers)], [np.log(2)*16*0.7, np.log(2)*16*0.7], color='gold', label='Entropy of 1 bit (optimal final distrubution)')

dropout = 0.1
#plot MI decay log(2)*(1-dropout)^l for each value of L 0 to 20
#plt.plot(np.arange(0, 20), [7*(1-dropout)**l for l in np.arange(0, 20)], color='red', label='1 bit through dropout channel (no error correction)')

"""
dropout=0.1
#plot the MI if encoding on 1 variable (dropout^L) for each value of L 0 to 20
plt.plot(np.arange(0, 20), [np.log(2)*(dropout**l) for l in np.arange(0, 20)], color='red', label='1 bit through dropout channel (no error correction)')

#plot the MI if 3 bit repetition coding (log(2) + log(2) + log(2)) for each value of L 0 to 20
print("a",len(np.arange(0, 20)))
print(len(list([max(3*(dropout**l), 1)*np.log(2) for l in np.arange(0, 20)])))
print("b")
plt.plot(np.arange(0, 20), list([min(dropout**l * 3, 1)*np.log(2) for l in np.arange(0, 20)]), color='green', label='3 bit reptition coding (no active error correction)')

#plot the MI if 16 bit repetition coding 16*log(2) for each value of L 0 to 20
plt.plot(np.arange(0, 20), [min(dropout**l * 16, 1)*np.log(2) for l in np.arange(0, 20)], color='blue', label='16 bit reptition coding (no active error correction)')

#plot the MI if 16 bit repetition coding with active error cancelation (1-dropout^16)^l 
print([min((1-dropout**16)**l, 1)*np.log(2) for l in np.arange(0, 20)])
plt.plot(np.arange(0, 20), [min(1-(1-dropout**16)**l, 1)*np.log(2)  for l in np.arange(0, 20)], color='purple', label='16 bit reptition coding (active error correction)')
"""


# give error bars as the variance of the rolling average
# following is deprecated
# plt.errorbar(np.arange(len(rolling_histories)), [rolling_histories[l]['val_loss'][-1:] for l in range(len(rolling_histories))], yerr=[np.sqrt(rolling_histories[l]['val_loss_var'][-1:]) for l in range(len(rolling_histories))], fmt='o')
#plt.errorbar(np.arange(len(rolling_histories)), [rolling_histories[l]['val_loss'][-1:] for l in range(len(rolling_histories))], yerr=[np.sqrt(rolling_histories[l]['val_loss_var'][-1:]) for l in range(len(rolling_histories))], fmt='o', capsize=5)
plt.title("Mutual Information between input and each layer")
plt.xlabel("Layer number")
plt.ylabel("Mutual Information")
#fix axis
plt.ylim(0, max([rolling_histories[l]['val_loss'][-1:] for l in range(len(rolling_histories))])*1.2)
plt.xlim(-0.2, len(model.layers)-1.8)
plt.show()