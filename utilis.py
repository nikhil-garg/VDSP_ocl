import nengo
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import tensorflow as tf
import os
from nengo.dists import Choice
from datetime import datetime
import pickle
from nengo.utils.matplotlib import rasterplot

plt.rcParams.update({'figure.max_open_warning': 0})
import time

from InputData import PresentInputWithPause
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev

# import nengo_ocl

from nengo.neurons import LIFRate
# from custom_rule import CustomRule
# from custom_rule import CustomRule_prev
from nengo.params import Parameter, NumberParam, FrozenObject
from nengo.dists import Choice, Distribution, get_samples, Uniform

from nengo.utils.numpy import clip, is_array_like

from nengo.connection import LearningRule
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import SimNeurons
from nengo.learning_rules import LearningRuleType
from nengo.builder.learning_rules import get_pre_ens,get_post_ens
from nengo.neurons import AdaptiveLIF
from nengo.synapses import Lowpass, SynapseParam
from nengo.params import (NumberParam,Default)
from nengo.dists import Choice
from nengo.utils.numpy import clip
import numpy as np
import random
import math


class MyLIF_in(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1 #reset voltage
        refractory_time[spiked_mask] = self.tau_ref + t_spike

def plan_MyLIF_in(
    queue,
    dt,
    J,
    V,
    W,
    outS,
    ref,
    tau,
    amp,
    N=None,
    tau_n=None,
    inc_n=None,
    upsample=1,
    fastlif=False,
    **kwargs
):
    adaptive = N is not None
    assert J.ctype == "float"
    for x in [V, W, outS]:
        assert x.ctype == J.ctype
    adaptive = False
    fastlif = False
    inputs = dict(J=J, V=V, W=W)
    outputs = dict(outV=V, outW=W, outS=outS)
    parameters = dict(tau=tau, ref=ref, amp=amp)
    if adaptive:
        assert all(x is not None for x in [N, tau_n, inc_n])
        assert N.ctype == J.ctype
        inputs.update(dict(N=N))
        outputs.update(dict(outN=N))
        parameters.update(dict(tau_n=tau_n, inc_n=inc_n))

    dt = float(dt)
    textconf = dict(
        type=J.ctype,
        dt=dt,
        upsample=upsample,
        adaptive=adaptive,
        dtu=dt / upsample,
        dtu_inv=upsample / dt,
        dt_inv=1 / dt,
        fastlif=fastlif,
    )
    decs = """
        char spiked;
        ${type} dV;
        const ${type} V_threshold = 1;
        const ${type} dtu = ${dtu}, dtu_inv = ${dtu_inv}, dt_inv = ${dt_inv};
% if adaptive:
        const ${type} dt = ${dt};
% endif
%if fastlif:
        const ${type} delta_t = dtu;
%else:
        ${type} delta_t;
%endif
        """
    # TODO: could precompute -expm1(-dtu / tau)
    text = """
        spiked = 0;
% for ii in range(upsample):
        W -= dtu;
% if not fastlif:
        delta_t = (W > dtu) ? 0 : (W < 0) ? dtu : dtu - W;
% endif
% if adaptive:
        dV = -expm1(-delta_t / tau) * (J - N - V);
% else:
        dV = -expm1(-delta_t / tau) * (J - V);
% endif
        V += dV;
% if fastlif:
        if (V < 0 || W > dtu)
            V = 0;
        else if (W >= 0)
            V *= 1 - W * dtu_inv;
% endif
        if (V > V_threshold) {
% if fastlif:
            const ${type} overshoot = dtu * (V - V_threshold) / dV;
            W = ref - overshoot + dtu;
% else:
            const ${type} t_spike = dtu + tau * log1p(
                -(V - V_threshold) / (J - V_threshold));
            W = ref + t_spike;
% endif
            V = 0;
            spiked = 1;
        }
% if not fastlif:
         else if (V < 0) {
            V = 0;
        }
% endif
% endfor
        outV = V;
        outW = W;
        outS = (spiked) ? amp*dt_inv : 0;
% if adaptive:
        outN = N + (dt / tau_n) * (inc_n * outS - N);
% endif
        """
    decs = as_ascii(Template(decs, output_encoding="ascii").render(**textconf))
    text = as_ascii(Template(text, output_encoding="ascii").render(**textconf))
    cl_name = "cl_alif" if adaptive else "cl_lif"
    return _plan_template(
        queue,
        cl_name,
        text,
        declares=decs,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        **kwargs,
    )


        
class MyLIF_out(LIFRate):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    initial_state : {str: Distribution or array_like}
        Mapping from state variables names to their desired initial value.
        These values will override the defaults set in the class's state attribute.
    """

    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None, inhib=[]
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage
        self.inhib = inhib

    def step(self, dt, J, output, voltage, refractory_time):
        # look these up once to avoid repeated parameter accesses

        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        output[:] = spiked_mask * (self.amplitude / dt)
        
        if(np.sum(output)!=0):
            voltage[voltage != np.max(voltage)] = 0 #WTA
            
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1
        refractory_time[spiked_mask] = self.tau_ref + t_spike

        


def fun_post(X,
       a1=0,a2=1,a3=0,a4=1,a5=1,#a6=1,a7=1,
       b1=1,b2=1,b3=1,b4=1,b5=1,#b6=1,b7=1,
       c1=0,c2=1,c3=0,c4=1,c5=1,#c6=1,c7=1,
       d1=1,d2=1,d3=1,d4=1,d5=1 ,#d6=1,d7=1,
       alpha1=1,alpha2=0    
       ): 
    w, vmem, vprog, vthp,vthn = X

    w_dep = w #Depression is dependent on w
    w_pot = 1-w #Potentiation is dependent on (1-w)
    
    v_ov_dep =  vmem - (vprog+vthp)
    v_ov_pot = (vprog-vthn) - vmem

    cond_dep = vmem>(vprog+vthp)
    cond_pot = vmem<(vprog-vthn)
    
    f_dep = a1 + a2*(w_dep) + a3*(w_dep*w_dep) + a4*(w_dep*w_dep*w_dep) + a5*(w_dep*w_dep*w_dep*w_dep) 
    f_pot = c1 + c2*(w_pot) + c3*(w_pot*w_pot) + c4*(w_pot*w_pot*w_pot) + c5*(w_pot*w_pot*w_pot*w_pot) 
    
    g_dep = d1 + d2*(v_ov_dep) + d3*(v_ov_dep*v_ov_dep) + d4*(v_ov_dep*v_ov_dep*v_ov_dep) + d5*(v_ov_dep*v_ov_dep*v_ov_dep*v_ov_dep)
    g_pot = b1 + b2*(v_ov_pot) + b3*(v_ov_pot*v_ov_pot)+ b4*(v_ov_pot*v_ov_pot*v_ov_pot) + b5*(v_ov_pot*v_ov_pot*v_ov_pot*v_ov_pot)
     
    dW = (abs(cond_pot*(alpha1*f_pot*g_pot)))  + (-1*(abs(cond_dep*(alpha2*f_dep*g_dep))))    
    return dW



popt = np.array((1.22495116e-02, -2.14968776e-01,  2.16351015e+00, -1.00745674e+00,
       -2.96338716e-01,  2.03309365e-03,  1.78418550e+00,  9.36936974e-01,
       -3.44177580e-02, -2.03283574e-01, -4.42570217e-03,  5.31904574e-01,
       -4.53948671e-01,  1.72725583e+00, -1.16844175e+00,  2.85775799e-03,
        1.80503076e+00,  8.02874904e-01,  5.23725555e-01, -5.77871444e-01,
        2.59452096e-01,  2.61974798e-01))



class CustomRule_post_v2(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.25,vthn=0.25):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(vmem, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            self.w = np.clip((self.w + dt*(fun_post((self.w,vmem, self.vprog, self.vthp,self.vthn),*popt)  -0)*post_out_matrix*self.lr), 0, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w
            
            return np.dot(self.w, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal


class CustomRule_post_v3(nengo.Process):
   
    def __init__(self, vprog=0,winit_min=0, winit_max=1, sample_distance = 1, lr=1,vthp=0.25,vthn=0.25, weight_quant =0):
       
        self.vprog = vprog  
        
        self.signal_vmem_pre = None
        self.signal_out_post = None

        self.winit_min = winit_min
        self.winit_max = winit_max
        
        
        self.sample_distance = sample_distance
        self.lr = lr
        self.vthp = vthp
        self.vthn = vthn
        self.weight_quant = weight_quant 

        
        self.history = [0]

        
        # self.tstep=0 #Just recording the tstep to sample weights. (To save memory)
        
        super().__init__()
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):  
       
        self.w = np.random.uniform(self.winit_min, self.winit_max, (shape_out[0], shape_in[0]))

        def step(t, x):

            assert self.signal_vmem_pre is not None
            assert self.signal_out_post is not None
            
            vmem = np.clip(self.signal_vmem_pre, -1, 1)
            
            post_out = self.signal_out_post
            
            vmem = np.reshape(vmem, (1, shape_in[0]))   

            post_out_matrix = np.reshape(post_out, (shape_out[0], 1))

            weight_quantization_matrix = ((np.random.randn(shape_out[0],shape_in[0])) -0.5)*self.weight_quant

            self.w = np.clip(((self.w + dt*(fun_post((self.w,vmem, self.vprog, self.vthp,self.vthn),*popt))*post_out_matrix*self.lr)+weight_quantization_matrix), 0, 1)
            
            # if (self.tstep%self.sample_distance ==0):
            #     self.history.append(self.w.copy())
            
            # self.tstep +=1
            self.history[0] = self.w.copy()
            # self.history.append(self.w.copy())
            # self.history = self.history[-2:]
            # self.history = self.w
            
            return np.dot(self.w, x)
        
        return step   

        # self.current_weight = self.w
    
    def set_signal_vmem(self, signal):
        self.signal_vmem_pre = signal
        
    def set_signal_out(self, signal):
        self.signal_out_post = signal




import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import (build_or_passthrough, get_post_ens,
                                          get_pre_ens)
from nengo.builder.operator import Copy, DotInc, Operator, Reset
from nengo.learning_rules import LearningRuleType, _remove_default_post_synapse
from nengo.params import Default, NdarrayParam, NumberParam
from nengo.synapses import Lowpass, SynapseParam


class VLR(LearningRuleType):
    """
    See the Nengo codebase
    (https://github.com/nengo/nengo/blob/master/nengo/learning_rules.py)
    for documentation and examples of how to construct this class, and what the super
    class constructor values are.
    """

    modifies = "weights"
    probeable = ("pre_voltages", "post_activities", "post_filtered","weights")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    vprog = NumberParam("vprog", readonly=True, default=-0.6)
    vthp = NumberParam("vthp", readonly=True, default=0.25)
    vthn = NumberParam("vthn", readonly=True, default=0.25)

    def __init__(
        self,
        learning_rate=Default,
        post_synapse=Default,
        vprog=Default,
        vthp=Default,
        vthn=Default
    ):
        super().__init__(learning_rate, size_in=0)
        self.post_synapse = post_synapse
        self.vprog = vprog
        self.vthp = vthp
        self.vthn = vthn


class SimVLR(Operator):
    """
    See the Nengo codebase
    (https://github.com/nengo/nengo/blob/master/nengo/builder/learning_rules.py)
    for the other examples of learning rule operators.
    """

    def __init__(self, pre_voltages, post_filtered,weights, delta, learning_rate,vprog,vthp,vthn, tag=None):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.vprog = vprog
        self.vthp = vthp
        self.vthn = vthn

        # Define what this operator sets, increments, reads and updates
        # See (https://github.com/nengo/nengo/blob/master/nengo/builder/operator.py)
        # for some other example operators
        self.sets = []
        self.incs = []
        self.reads = [pre_voltages, post_filtered, weights]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_voltages(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]
    
    @property
    def weights(self):
        return self.reads[2]

    @property
    def _descstr(self):
        return f"pre={self.pre_voltages}, post={self.post_filtered} -> {self.delta}"

    def make_step(self, signals, dt, rng):
        # Get signals from model signal dictionary
        pre_voltages = signals[self.pre_voltages]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        weights = signals[self.weights]
        
        pre_voltages = np.reshape(pre_voltages, (1, delta.shape[1]))
        post_filtered = np.reshape(post_filtered, (delta.shape[0], 1))

        def step_vlr():
            # Put learning rule logic here
            
            delta[...] = post_filtered*dt*fun_post((weights,pre_voltages,self.vprog,self.vthp,self.vthn),*popt)*self.learning_rate

        return step_vlr


@Builder.register(VLR)  # Register the function below with the Nengo builder
def build_vlr(model, vlr, rule):
    """
    See the Nengo codebase
    (https://github.com/nengo/nengo/blob/master/nengo/builder/learning_rules.py#L594)
    for the documentation for this function.
    """

    # Extract necessary signals and objects from the model and learning rule
    conn = rule.connection
    pre_voltages = model.sig[get_pre_ens(conn).neurons]["voltage"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    weights = model.sig[conn]["weights"]
    
    post_filtered = build_or_passthrough(model, vlr.post_synapse, post_activities)
#     post_filtered = post_activities

    # Instantiate and add the custom learning rule operator to the Nengo model op graph
    model.add_op(
        SimVLR(
            pre_voltages,
            post_filtered,
            weights,
            model.sig[rule]["delta"],
            learning_rate=vlr.learning_rate,
            vprog = vlr.vprog,
            vthp = vlr.vthp,
            vthn = vlr.vthn
            
        )
    )

    # Expose these signals for probes
    model.sig[rule]["pre_voltages"] = pre_voltages
    model.sig[rule]["post_activities"] = post_activities
    model.sig[rule]["post_filtered"] = post_filtered


#create new neuron type STDPLIF 
def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)

#---------------------------------------------------------------------
# Neuron Model declaration 
#---------------------------------------------------------------------

#create new neuron type STDPLIF 

class STDPLIF(AdaptiveLIF):
    probeable = ('spikes', 'voltage', 'refractory_time','adaptation','inhib') #,'inhib'
    
    def __init__(self, spiking_threshold = 70, inhibition_time=10,inhib=[],T = 0.0, **lif_args): # inhib=[],T = 0.0
        super(STDPLIF, self).__init__(**lif_args)
        # neuron args (if you have any new parameters other than gain
        # an bais )
        self.inhib = inhib
        self.T = T
        self.spiking_threshold=spiking_threshold
        self.inhibition_time=inhibition_time
    @property
    def _argreprs(self):
        args = super(STDPLIF, self)._argreprs
        print("argreprs")
        return args

    # dt : timestamps 
    # J : Input currents associated with each neuron.
    # output : Output activities associated with each neuron.
    def step(self, dt, J, output, voltage, refractory_time, adaptation,inhib):#inhib

        self.T += dt
        
        # if(np.max(J) !=0):
        #     J = np.divide(J,np.max(J)) * 2

        n = adaptation
        
        J = J - n
        # ----------------------------

        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = (voltage > self.spiking_threshold)
        output[:] = spiked_mask * (self.amplitude / dt)
        output[voltage != np.max(voltage)] = 0  
        if(np.sum(output) != 0):
            voltage[voltage != np.max(voltage)] = 0 
            inhib[(voltage != np.max(voltage)) & (inhib == 0)] = self.inhibition_time/(dt*1000)
        #print("voltage : ",voltage)
        voltage[inhib != 0] = 0 
        J[inhib != 0] = 0
        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = -1 #Reset voltage
        voltage[refractory_time > 0] = -1 #Refractory voltage
        refractory_time[spiked_mask] = self.tau_ref + t_spike
        # ----------------------------

        n += (dt / self.tau_n) * (self.inc_n * output - n)

        #AdaptiveLIF.step(self, dt, J, output, voltage, refractory_time, adaptation)
        inhib[inhib != 0] += - 1
        #J[...] = 0
        #output[...] = 0
        

#---------------------------------------------------------------------
#add builder for STDPLIF
#---------------------------------------------------------------------

@Builder.register(STDPLIF)
def build_STDPLIF(model, STDPlif, neurons):
    
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['pre_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.pre_filtered" % neurons)
    model.sig[neurons]['post_filtered'] = Signal(
        np.zeros(neurons.size_in), name="%s.post_filtered" % neurons)
    model.sig[neurons]['inhib'] = Signal(
        np.zeros(neurons.size_in), name="%s.inhib" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in),name= "%s.adaptation" % neurons
    )
    # set neuron output for a given input
    model.add_op(SimNeurons(neurons=STDPlif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            state={"voltage": model.sig[neurons]['voltage'],
                                    "refractory_time": model.sig[neurons]['refractory_time'],
                                    "adaptation": model.sig[neurons]['adaptation'],
                                    "inhib": model.sig[neurons]['inhib']
                                     }))


import os
import re
# import cv2


def gen_video(directory, f_prename):
    
    assert os.path.exists(directory)

    img_array = []
    for filename in os.listdir(directory):
        if f_prename in filename:
            nb = re.findall(r"(\d+).png", filename)
            if len(nb) == 1:
                img = cv2.imread(os.path.join(directory, filename))
                img_array.append((int(nb[0]), img))

    height, width, layers = img.shape
    size = (width, height)

    img_array = sorted(img_array, key=lambda x: x[0])
    video_path = os.path.join(directory, f"{f_prename}.avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), 2, size)

    for _, img in img_array:
        out.write(img)
    out.release()

    print(f"{video_path} generated successfully.")