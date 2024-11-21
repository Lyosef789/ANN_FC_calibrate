#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

tfd = tfp.distributions

def prior_trainable(kernel_size, bias_size=0, dtype=None):
    """
    Define the prior for Bayesian layers.
    """
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1)),
    ])

def posterior(kernel_size, bias_size=0, dtype=None):
    """
    Define the posterior for Bayesian layers.
    """
    n = kernel_size + bias_size
    c = tf.math.log(tf.math.expm1(1.0))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)),
    ])

def create_bnn_model(train_size):
    """
    Create a Bayesian Neural Network model.
    """
    inputs = layers.Input(shape=(50,))  # Assuming input features are 50
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tfp.layers.DenseVariational(8, make_prior_fn=prior_trainable, make_posterior_fn=posterior, kl_weight=1/train_size)(x)
    x = layers.Dense(2)(x)
    outputs = tfp.layers.IndependentNormal(1)(x)
    return tf.keras.Model(inputs, outputs)

def improved_penalized_nll(targets, estimated_distribution, max_value):
    """
    Penalized Negative Log-Likelihood (NLL) Loss Function.
    Dynamically adjusts penalty scaling based on max value.
    """
    base_nll = -estimated_distribution.log_prob(targets)
    mean_predictions = estimated_distribution.mean()

    penalty = tf.where(
        mean_predictions < targets,
        tf.square(mean_predictions - targets),
        tf.zeros_like(base_nll)
    )

    # Normalize penalty by max_value to handle dynamic scaling
    scaled_penalty = penalty / max_value
    return tf.reduce_mean(base_nll + scaled_penalty)

