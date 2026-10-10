import numpy as np


def flops(I, B):
    return B * (17779 * I**2 + 313600)

def memory(I, B):

    model_memory = 4_174_632

    activations = 4 * B * (33 * I**2 + 868)

    return model_memory + activations

def bytes_moved(I, B):
    return 4 * (
        105 * B * I**2
        + 2148 * B
        + 1_043_652
    )

def latency(I, B, theta):
    F = flops(I, B)
    Q = bytes_moved(I, B)

    t0 = theta["t0"]
    alpha = theta["alpha"]
    beta = theta["beta"]

    return t0 + np.maximum(
        alpha * F,
        beta * Q,
    )

def energy(I, B, theta_energy):
    F = flops(I, B)
    Q = bytes_moved(I, B)

    e0 = theta_energy["e0"]
    e_flop = theta_energy["e_flop"]
    e_byte = theta_energy["e_byte"]

    return (
        e0
        + e_flop * F
        + e_byte * Q
    )
