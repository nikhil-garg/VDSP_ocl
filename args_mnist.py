# -*- coding: utf-8 -*-
"""
Created on 4th Jan 2021

@author: Nikhil
"""

import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a VDSP based MNIST classifier"
    )

    # Defining the model
    parser.add_argument(
        "--input_nbr",
        default=60000, 
        type=int, 
        help="Number of images to consider for training"
    )

    parser.add_argument(
        "--dt",
        default=0.005, 
        type=float, 
        help="Time step"
    )
    parser.add_argument(
        "--digit",
        default=4,
        type=int,
        help="The digit to consider for geting receptive field",
    )
    parser.add_argument(
        "--presentation_time",
        default=0.35,
        type=float,
        help="Presentation time of one image",
    )
    parser.add_argument(
        "--pause_time",
        default=0,
        type=float,
        help="Pause time",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=float,
        help="Number of iterations to train for",
    )

    parser.add_argument(
        "--n_in",
        default=784,
        type=int,
        help="Number of input neurons",
    )

    parser.add_argument(
        "--amp_neuron",
        default=0.5607848,
        type=float,
        help="Transform from synapse to output neurons"
    )
    parser.add_argument(
        "--n_neurons",
        default=30,
        type=float,
        help="Number of output neurons",
    )
    parser.add_argument(
        "--tau_in",
        default=0.06,
        type=float,
        help="Leak constant of input neurons",
    )

    parser.add_argument(
        "--tau_out",
        default=0.06,
        type=float,
        help="Leak constant of output neurons",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="Learning rate of VDSP",
    )
    parser.add_argument(
        "--filename",
        default="default",
        type=str,
        help="filename of final weights",
    )

    parser.add_argument(
        "--vprog",
        default=-0.75,
        type=float,
        help="vprog",
    )
    parser.add_argument(
        "--rate_out",
        default=20,
        type=float,
        help="Firing rate for output neuron",
    )
    parser.add_argument(
        "--rate_in",
        default=20,
        type=float,
        help="Firing rate for input neuron",
    )

    parser.add_argument(
        "--gain_out",
        default=2,
        type=float,
        help="gain for output neuron",
    )
    parser.add_argument(
        "--gain_in",
        default=2,
        type=float,
        help="gain for input neuron",
    )

    parser.add_argument(
        "--bias_out",
        default=0,
        type=float,
        help="bias for output neuron",
    )
    parser.add_argument(
        "--bias_in",
        default=0,
        type=float,
        help="bias for input neuron",
    )

    parser.add_argument(
        "--thr_out",
        default=1,
        type=float,
        help="Threshold of output layer",
    )
    parser.add_argument(
        "--inhibition_time",
        default=10,
        type=float,
        help="inhibition_time",
    )
    parser.add_argument(
        "--var_ratio",
        default=0,
        type=float,
        help="Variability of vth. Between 0 and 1",
    )
    parser.add_argument(
        "--vthp",
        default=0.16,
        type=float,
        help="Switching threshold of memristor",
    )
    parser.add_argument(
        "--vthn",
        default=0.15,
        type=float,
        help="Switching threshold of memristor",
    )
    parser.add_argument(
        "--weight_quant",
        default=0,
        type=float,
        help="Variability of weight update",
    )
    parser.add_argument(
        "--amp_var",
        default=0,
        type=float,
        help="Variability of Ap and An in VDSP",
    )
    parser.add_argument(
        "--dw_var",
        default=0,
        type=float,
        help="Variability of dW",
    )
    parser.add_argument(
        "--g_var",
        default=0,
        type=float,
        help="Variability of gmax and gmin",
    )

    parser.add_argument(
        "--gmax",
        default=0.0085,
        type=float,
    )
    parser.add_argument(
        "--gmin",
        default=0.0000085,
        type=float,
    )

    parser.add_argument(
        "--log_file_path",
        default=None,
        type=str,
        help="log file path",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=float,
        help="Seed of random number generator",
    )

    my_args = parser.parse_args()

    return my_args
