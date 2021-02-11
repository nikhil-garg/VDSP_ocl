import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import nengo
import numpy as np
from numpy import random
#from src.Models.Neuron.STDPLIF import STDPLIF
#from DataLog import DataLog
from InputData import PresentInputWithPause
# from Heatmap import AllHeatMapSave,HeatMapSave
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
from utilis import *
import pickle
import tensorflow as tf

#############################
# load the data
#############################

img_rows, img_cols = 28, 28
input_nbr = 6000
probe_sample_rate = (input_nbr/10)/1000 #Probe sample rate. Proportional to input_nbr to scale down sampling rate of simulations 

Dataset = "Mnist"
# (image_train, label_train), (image_test, label_test) = load_mnist()
(image_train, label_train), (image_test, label_test) = (tf.keras.datasets.mnist.load_data())


#select the 0s and 1s as the two classes from MNIST data
image_test_filtered = []
label_test_filtered = []

for i in range(0,input_nbr):
#  if (label_train[i] == 1 or label_train[i] == 0):
        image_test_filtered.append(image_test[i])
        label_test_filtered.append(label_test[i])

print("actual input",len(label_test_filtered))
print(np.bincount(label_test_filtered))

image_test_filtered = np.array(image_test_filtered)
label_test_filtered = np.array(label_test_filtered)

#############################

model = nengo.Network(label="My network",)



presentation_time = 0.35 #0.35
pause_time = 0 #0.15
#input layer
n_in = 784
n_neurons = 20

# Learning params



with model:
    # input layer 
      # picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
    picture = nengo.Node(nengo.processes.PresentInput(image_train_filtered, presentation_time=presentation_time))
    true_label = nengo.Node(nengo.processes.PresentInput(label_train_filtered, presentation_time=presentation_time))
        # true_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))
    input_layer = nengo.Ensemble(
        n_in,
        1,
        label="Input",
        neuron_type=MyLIF_in(tau_rc=0.3,min_voltage=-1,amplitude=0.3),#nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2)),#nengo.LIF(amplitude=0.2),# nengo.neurons.PoissonSpiking(nengo.LIFRate(amplitude=0.2))
        gain=nengo.dists.Choice([2]),
        encoders=nengo.dists.Choice([[1]]),
        bias=nengo.dists.Choice([0]))

    input_conn = nengo.Connection(picture,input_layer.neurons,)

    # weights randomly initiated 
    #layer1_weights = np.round(random.random((n_neurons, 784)),5)
    # define first layer
    layer1 = nengo.Ensemble(
         n_neurons,
         1,
         label="layer1",
         neuron_type=STDPLIF(tau_rc=0.3, min_voltage=-1),
         intercepts=nengo.dists.Choice([0]),
         max_rates=nengo.dists.Choice([20,20]),
         encoders=nengo.dists.Choice([[1]]))

    # w = nengo.Node(CustomRule_post_v2(**learning_args), size_in=784, size_out=n_neurons)
    
    nengo.Connection(input_layer.neurons, layer1.neurons,transform=0)
 
   
    p_true_label = nengo.Probe(true_label, sample_every=probe_sample_rate)
    p_layer_1 = nengo.Probe(layer1.neurons, sample_every=probe_sample_rate)
    #if(not full_log):
    #    nengo.Node(log)

    #############################

step_time = (presentation_time + pause_time) 

with nengo.Simulator(model,dt=0.005) as sim:
    
    
    sim.run(step_time * label_test_filtered.shape[0])




# pickle.dump(weights, open( "mnist_params_STDP", "wb" ))
neuron_class = np.array( [[3.]
 [1.]
 [1.]
 [5.]
 [5.]
 [7.]
 [0.]
 [2.]
 [7.]
 [7.]
 [2.]
 [6.]
 [0.]
 [9.]
 [8.]
 [6.]
 [4.]
 [1.]
 [9.]
 [2.]])


t_data = sim.trange(sample_every=probe_sample_rate)

labels = sim.data[p_true_label][:,0]

output_spikes = sim.data[p_layer_1]

n_classes = 10

rate_data = nengo.synapses.Lowpass(0.1).filtfilt(sim.data[p_layer_1])

predicted_labels = labels * 0   

correct_classified = 0
wrong_classified = 0

for t in range(len(t_data)):
    if(labels[t]>0):
    # Find the index of neuron with highest firing rate : k
        k = np.argmax(rate_data[t])
        predicted_labels[t] = neuron_class[k]
        if(predicted_labels[t] == labels[t]):
            correct_classified+=1
        else:
            wrong_classified+=1
        
accuracy = correct_classified/ (correct_classified+wrong_classified)*100
print("Accuracy: ", accuracy)