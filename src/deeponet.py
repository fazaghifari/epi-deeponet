import math
import joblib
import tf_keras as keras

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tf_keras.models import Model
from tf_keras.layers import Input, Lambda, Dense
import tensorflow_probability as tfp

from tqdm import tqdm
from matplotlib import pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

def build_mlp(inputs, output_features: int, hidden_features: int, 
              num_hidden_layers: int, bs:int, modelname="Trunk"):
    """
    Builds an MLP regression model.
    
    Parameters:
        x: Input tensor.
        output_features (int): Number of output features.
        hidden_features (int): Number of units in each hidden layer.
        num_hidden_layers (int): Number of hidden layers.
        bs (int): Batch size
    
    Returns:
        array: keras tensor.
    """
    assert num_hidden_layers > 1, "Number of the hidden layers must be greater than 1"

    x = inputs
    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (bs * 1.0)
    bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (bs * 1.0)
    
    for ii in range(num_hidden_layers-1):
        x = tfp.layers.DenseFlipout(hidden_features, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn, activation='relu', name=f"Hidden_{modelname}_{ii}")(x)
    
    # Output layer with linear activation for regression
    outputs = tfp.layers.DenseFlipout(output_features, bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn, activation='relu', name=f"Out_{modelname}_{ii}")(x)
    
    return outputs


def fn(x):
    y = tf.einsum("ij, ij->i", x[0], x[1])
    y = tf.expand_dims(y, axis = 1)
    return y

def normal_sp(params):
    return tfd.Normal(loc = params[:, 0:1], scale = 0.001+tf.math.softplus(params[:, 1:2]))   

def deeponet(y_in_size, u_in_size, trunk_params, branch_params, batch_size, out_feat=1):
    """_summary_

    Args:
        y_in_size (tuple): spatial input size
        u_in_size (tuple): function input size
        trunk_params (dict): trunk net parameters
        branch_params (dict): branch net parameters
        out_feat (int, optional): number of output feature. Defaults to 1.

    Returns:
        keras.Model: output model
    """
    y_input = Input(y_in_size, name="Input_Trunk")
    u_input = Input(u_in_size, name="Input_Branch")

    # Trunk layer
    trunk_out = build_mlp(y_input, trunk_params["output_features"], trunk_params["hidden_features"], 
                          trunk_params["num_hidden_layers"], batch_size, modelname="Trunk")
    
    # Branch layer
    branch_out = build_mlp(u_input, branch_params["output_features"], branch_params["hidden_features"], 
                           branch_params["num_hidden_layers"], batch_size, modelname="Branch")

    # Multiply trunk and branch
    mult_out = Lambda(fn, output_shape = [None, 1])([branch_out, trunk_out])
    out = tfp.layers.DenseFlipout(2)(mult_out)

    dist = tfp.layers.DistributionLambda(normal_sp)(out)

    model = Model(inputs=[y_input, u_input], outputs=dist, name="VB-DeepONet")

    return model