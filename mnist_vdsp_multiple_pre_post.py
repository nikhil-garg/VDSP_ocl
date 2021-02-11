
import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import os
from nengo.dists import Choice
from datetime import datetime
from nengo_extras.data import load_mnist
import pickle
from nengo.utils.matplotlib import rasterplot

import time

from InputData import PresentInputWithPause

from nengo_extras.graphviz import net_diagram

from nengo.neurons import LIFRate

from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform

from nengo.utils.numpy import clip, is_array_like
from utilis import *


from args_mnist import args as my_args
import itertools
import random
import logging



def fun_pre(X,
       a1=0,a2=1,a3=0,
       b1=1,b2=1,b3=1,b4=1,b5=1,b6=1,b7=1,
       c1=0,c2=1,c3=0,
       d1=1,d2=1,d3=1,d4=1,d5=1,d6=1,d7=1,
       e1=0, e2=0, e3=0, e4=0,e5=0,e6=0,
       alpha1=1,alpha2=0    
       ): 
    w, vmem = X
    vthp=0.25
    vthn=0.25
    vprog=0
    w_pos = (e1*w) + e3
    w_neg = e2*(1-w) + e4
    v_ov_p =  vmem - (vprog+vthp) + e5
    v_ov_n = (vprog-vthn) - vmem  + e6
    cond_1 = vmem<(vprog-vthn)
    cond_2 = vmem>(vprog+vthp)
    f1 = a1 + a2*(w_pos**1) + a3*(w_pos**2)
    g1 = b1 + b2*np.sin(b3*v_ov_n + b4) + b5*np.cos(b6*v_ov_n + b7)
    f2 = c1 + c2*(w_neg**1) + c3*(w_neg**2)
    g2 = d1 + d2*np.sin(d3*v_ov_p + d4) + d5*np.cos(d6*v_ov_p + d7)
    dW = (-1*abs(cond_1*(alpha1*f1*g1)))  + (cond_2*(alpha2*cond_2*f2*g2))    
    return dW

def fun_post(X,
       a1=0,a2=1,a3=0,
       b1=1,b2=1,b3=1,b4=1,b5=1,b6=1,b7=1,
       c1=0,c2=1,c3=0,
       d1=1,d2=1,d3=1,d4=1,d5=1,d6=1,d7=1,
       e1=0, e2=0, e3=0, e4=0,e5=0,e6=0,
       alpha1=1,alpha2=0    
       ): 
    w, vmem = X
    vthp=0.25
    vthn=0.25
    vprog=0
    w_pos = (e1*w) + e3
    w_neg = e2*(1-w) + e4
    v_ov_p =  vmem - (vprog+vthp) + e5
    v_ov_n = (vprog-vthn) - vmem  + e6
    cond_1 = vmem<(vprog-vthn)
    cond_2 = vmem>(vprog+vthp)
    f1 = a1 + a2*(w_pos**1) + a3*(w_pos**2)
    g1 = b1 + b2*np.sin(b3*v_ov_n + b4) + b5*np.cos(b6*v_ov_n + b7)
    f2 = c1 + c2*(w_neg**1) + c3*(w_neg**2)
    g2 = d1 + d2*np.sin(d3*v_ov_p + d4) + d5*np.cos(d6*v_ov_p + d7)
    dW = (abs(cond_1*(alpha1*f1*g1)))  + (-1*cond_2*(alpha2*cond_2*f2*g2))    
    return dW
# ydata = np.array(delta_state_list)

popt_post = np.array((-5.44634746e+00, -8.42848816e-01,  1.49956029e+00,  1.98056395e+00,
        4.33050527e+00,  5.28219321e-01,  7.24397333e-02,  3.37358302e+00,
        9.98808901e-01,  2.87121896e+00, -5.57633406e-01, -2.75112832e-01,
        1.60193659e+00,  4.09073550e-01, -9.26010737e-01,  4.91251299e-01,
        6.61539169e-03, -1.05305318e-01,  1.93590366e+00,  3.55720979e-01,
        3.61854190e-03, -3.54039473e-01, -1.64873794e+00, -1.93935931e-01,
        1.14033130e+00,  4.57240635e-01,  5.57668985e+00,  2.64857548e+00))

popt_pre = np.array((-1.20848198,  2.49846595, -0.28987743, -0.14117428,  1.61824135,
        0.93103767, -0.37808845,  0.48727376,  1.78462442,  1.01036057,
       -1.09633602,  2.2774841 ,  0.03198922, -0.08799407,  0.371004  ,
        1.97859903,  0.32183791,  1.69286796,  0.93152593,  1.49844153,
        2.41056729,  2.82007663,  0.24099919,  0.33549382,  0.16336168,
        0.71717484, -0.11953137, -0.04973603))

class CustomRule_pre_post(nengo.Process):
   
    def __init__(self, vthp=0.25, vthn=0.25, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1):
       
        self.vthp = vthp
        self.vthn = vthn
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_vmem_post = None
        self.signal_out_post = None
        
        self.sample_distance = sample_distance
        self.lr = lr
        
        self.history = []
        self.current_weight = []
        self.update_history = []
        
        self.vmem_prev_pre = 0
        
        self.winit_min = winit_min
        self.winit_max = winit_max
        
        self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))
        dw = np.zeros((shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_vmem_post is not None
            assert self.signal_out_post is not None
            
            vmem_pre = self.signal_vmem_pre
            vmem_pre = np.clip(vmem_pre, -1, 1)
            vmem_pre = np.reshape(vmem_pre, (1, shape_in[0])) 
            vmem_pre = np.vstack([vmem_pre]*shape_out[0])   
            
            vmem_post = self.signal_vmem_post
            vmem_post = np.reshape(vmem_post, (shape_out[0],1))
            vmem_post = np.clip(vmem_post, -1, 1)
            vmem_post = np.hstack([vmem_post]*shape_in[0])
            
               
            pre_out_matrix = np.reshape(x, (1, shape_in[0]))         
            pre_out_matrix = np.vstack([pre_out_matrix]*shape_out[0])    
            
            post_out = self.signal_out_post        
            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))
            post_out_matrix = np.hstack([post_out_matrix]*shape_in[0])
                     
            
            dw_post = post_out_matrix*dt*fun_post((self.w,vmem_pre),*popt_post)   
            dw_pre = pre_out_matrix*dt*fun_pre((self.w,vmem_post),*popt_pre) 
            
            self.w += (dw_post+dw_pre)*self.lr  
            self.w = np.clip(self.w, 0,1)            
            
            if (self.tstep%self.sample_distance ==0):
                self.history.append(self.w.copy())
                self.update_history.append(dw.copy())
            
            self.tstep +=1
            
            self.vmem_prev_pre = vmem_pre.copy()
            return np.dot(self.w, x)
        
        return step   

        self.current_weight = self.w
    
    def set_signal_vmem_pre(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_vmem_post(self, signal):
        self.signal_vmem_post = signal
        
    def set_signal_out_post(self, signal):
        self.signal_out_post = signal





def evaluate_mnist_multiple(args):

    #############################
    # load the data
    #############################
    input_nbr = args.input_nbr

    (image_train, label_train), (image_test, label_test) = (tf.keras.datasets.mnist.load_data())

    probe_sample_rate = (input_nbr/10)/1000 #Probe sample rate. Proportional to input_nbr to scale down sampling rate of simulations 
    # probe_sample_rate = 1000
    image_train_filtered = []
    label_train_filtered = []

    x = args.digit

    for i in range(0,input_nbr):
      
        image_train_filtered.append(image_train[i])
        label_train_filtered.append(label_train[i])

    image_train_filtered = np.array(image_train_filtered)
    label_train_filtered = np.array(label_train_filtered)


    #Simulation Parameters 
    #Presentation time
    presentation_time = args.presentation_time #0.20
    #Pause time
    pause_time = args.pause_time
    #Iterations
    iterations=args.iterations
    #Input layer parameters
    n_in = args.n_in
    # g_max = 1/784 #Maximum output contribution
    g_max = args.g_max
    n_neurons = args.n_neurons # Layer 1 neurons
    inhib_factor = args.inhib_factor #Multiplication factor for lateral inhibition


    input_neurons_args = {
            "n_neurons":n_in,
            "dimensions":1,
            "label":"Input layer",
            "encoders":nengo.dists.Uniform(1,1),
            # "max_rates":nengo.dists.Uniform(22,22),
            # "intercepts":nengo.dists.Uniform(0,0),
            "gain":nengo.dists.Uniform(2,2),
            "bias":nengo.dists.Uniform(0,0),
            "neuron_type":MyLIF_in(tau_rc=args.tau_in,min_voltage=-1)
            # "neuron_type":nengo.neurons.SpikingRectifiedLinear()#SpikingRelu neuron. 
    }

    #Layer 1 parameters
    layer_1_neurons_args = {
            "n_neurons":n_neurons,
            "dimensions":1,
            "label":"Layer 1",
            "encoders":nengo.dists.Uniform(1,1),
            # "gain":nengo.dists.Uniform(2,2),
            # "bias":nengo.dists.Uniform(0,0),
            "intercepts":nengo.dists.Choice([0]),
            "max_rates":nengo.dists.Choice([20,20]),
            # "noise":nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 0.5), seed=1), 
            # "neuron_type":nengo.neurons.LIF(tau_rc=args.tau_out, min_voltage=0)
            # "neuron_type":MyLIF_out(tau_rc=args.tau_out, min_voltage=-1)
            # "neuron_type":MyLIF_in(tau_rc=args.tau_out,min_voltage=-1)

            "neuron_type":STDPLIF(tau_rc=args.tau_out, min_voltage=-1),
    }

    # "noise":nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 20), seed=1),     

    #Lateral Inhibition parameters
    lateral_inhib_args = {
            "transform": inhib_factor* (np.full((n_neurons, n_neurons), 1) - np.eye(n_neurons)),
            "synapse":args.inhib_synapse,
            "label":"Lateral Inhibition"
    }

    #Learning rule parameters
    learning_args = {
            "lr": args.lr,
            "winit_min":0,
            "winit_max":0.1,
    #         "tpw":50,
    #         "prev_flag":True,
            "sample_distance": int((presentation_time+pause_time)*200), #Store weight after 10 images
    }

    argument_string = "presentation_time: "+ str(presentation_time)+ "\n pause_time: "+ str(pause_time)+ "\n input_neurons_args: " + str(input_neurons_args)+ " \n layer_1_neuron_args: " + str(layer_1_neurons_args)+"\n Lateral Inhibition parameters: " + str(lateral_inhib_args) + "\n learning parameters: " + str(learning_args)+ "\n g_max: "+ str(g_max) 

    images = image_train_filtered
    labels = label_train_filtered


    model = nengo.Network("My network")
    #############################
    # Model construction
    #############################
    with model:
        picture = nengo.Node(PresentInputWithPause(images, presentation_time, pause_time,0))
        # picture = nengo.Node(nengo.processes.PresentInput(images, presentation_time=presentation_time))
        # true_label = nengo.Node(nengo.processes.PresentInput(labels, presentation_time=presentation_time))
        strue_label = nengo.Node(PresentInputWithPause(labels, presentation_time, pause_time,-1))

        # input layer  
        input_layer = nengo.Ensemble(**input_neurons_args)
        input_conn = nengo.Connection(picture,input_layer.neurons,synapse=None)

        #first layer
        layer1 = nengo.Ensemble(**layer_1_neurons_args)

        #Weights between input layer and layer 1
        w = nengo.Node(CustomRule_pre_post(**learning_args), size_in=n_in, size_out=n_neurons)
        nengo.Connection(input_layer.neurons, w, synapse=None)
        nengo.Connection(w, layer1.neurons,transform=1/args.g_max, synapse=None)
        # nengo.Connection(w, layer1.neurons,transform=g_max, synapse=None)

        #Lateral inhibition
        inhib = nengo.Connection(layer1.neurons,layer1.neurons,**lateral_inhib_args) 

        #Probes
        # p_true_label = nengo.Probe(true_label, sample_every=probe_sample_rate)
        p_input_layer = nengo.Probe(input_layer.neurons, sample_every=probe_sample_rate)
        p_layer_1 = nengo.Probe(layer1.neurons, sample_every=probe_sample_rate)
        weights = w.output.history

        


    # with nengo_ocl.Simulator(model) as sim :   
    with nengo.Simulator(model, dt=0.005) as sim:

        w.output.set_signal_vmem_pre(sim.signals[sim.model.sig[input_layer.neurons]["voltage"]])
        w.output.set_signal_vmem_post(sim.signals[sim.model.sig[layer1.neurons]["voltage"]])
        w.output.set_signal_out_post(sim.signals[sim.model.sig[layer1.neurons]["out"]])
                    
        sim.run((presentation_time+pause_time) * labels.shape[0]*iterations)

    #save the model
    # now = time.strftime("%Y%m%d-%H%M%S")
    # folder = os.getcwd()+"/MNIST_VDSP"+now
    # os.mkdir(folder)
    last_weight = weights[-1]

    # pickle.dump(weights, open( folder+"/trained_weights", "wb" ))
    # pickle.dump(argument_string, open( folder+"/arguments", "wb" ))

    sim.close()

    return weights, sim.data[p_input_layer], sim.data[p_layer_1], sim.trange(sample_every=probe_sample_rate)


    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     print(tstep)
    #     fig, axes = plt.subplots(1,1, figsize=(3,3))

    #     for i in range(0,(n_neurons)):
            
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot()
    #         cax = ax1.matshow(np.reshape(weights[tstep][i],(28,28)),interpolation='nearest', vmax=1, vmin=0)
    #         fig.colorbar(cax)

    #     plt.tight_layout()    
    #     fig.savefig(folder+'/weights'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "weights")



    # for tstep in np.arange(0, len(weights), 1):
    #     tstep = int(tstep)
    #     print(tstep)
    #     fig, axes = plt.subplots(1,1, figsize=(3,3))

    #     for i in range(0,(n_neurons)):
            
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot()
    #         cax = ax1.hist(weights[tstep][i])
    #         ax1.set_xlim(0,1)
    #         ax1.set_ylim(0,350)

    #     plt.tight_layout()    
    #     fig.savefig(folder+'/histogram'+str(tstep)+'.png')
    #     plt.close('all')

    # gen_video(folder, "histogram")



if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()


    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    # Fix the seed of all random number generator
    seed = 500
    random.seed(seed)
    np.random.seed(seed)

    weights, p_input_layer, p_layer_1, time_points = evaluate_mnist_multiple(args)

    now = time.strftime("%Y%m%d-%H%M%S")
    folder = os.getcwd()+"/MNIST_VDSP"+now
    os.mkdir(folder)


    plt.figure(figsize=(12,10))

    plt.subplot(2, 1, 1)
    plt.title('Input neurons')
    rasterplot(time_points, p_input_layer)
    plt.xlabel("Time [s]")
    plt.ylabel("Neuron index")

    plt.subplot(2, 1, 2)
    plt.title('Output neurons')
    rasterplot(time_points, p_layer_1)
    plt.xlabel("Time [s]")
    plt.ylabel("Neuron index")

    plt.tight_layout()

    plt.savefig(folder+'/raster'+'.png')


    for tstep in np.arange(0, len(weights), 10):
        tstep = int(tstep)
        # tstep = len(weights) - tstep -1


        print(tstep)

        columns = int(args.n_neurons/5)
        fig, axes = plt.subplots(int(args.n_neurons/columns), int(columns), figsize=(20,25))

        for i in range(0,(args.n_neurons)):

            axes[int(i/columns)][int(i%columns)].matshow(np.reshape(weights[tstep][i],(28,28))) #,interpolation='nearest', vmax=1, vmin=0)


        plt.tight_layout()    
        fig.savefig(folder+'/weights'+str(tstep)+'.png')
        plt.close('all')

    gen_video(folder, "weights")



    logger.info('All done.')