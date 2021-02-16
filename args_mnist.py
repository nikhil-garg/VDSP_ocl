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
        "--g_max",
        default=0.3,
        type=float,
        help="Transform from synapse to output neurons"
    )
    parser.add_argument(
        "--n_neurons",
        default=20,
        type=float,
        help="Number of output neurons",
    )
    parser.add_argument(
        "--tau_in",
        default=0.3,
        type=float,
        help="Leak constant of input neurons",
    )

    parser.add_argument(
        "--tau_out",
        default=0.3,
        type=float,
        help="Leak constant of output neurons",
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
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
        default=-0.6,
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

    my_args = parser.parse_args()

    return my_args
