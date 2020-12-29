
import tensorflow as tf
import numpy as np

"""
Multipoint precision idea:

Instead of returning a tensor, return three tensors:
(std, tiny, factor)

Where the result is equal to std + tiny * factor
"""

@tf.function
def rolling_diff(fn, boundary_values):
    """
    Compute f(x+h)-f(x).
    """
    shape = tf.shape(fn)
    fn_size = shape[0]
    f_x = tf.slice(fn, (0,), (fn_size-2,))
    f_x_h = tf.slice(fn, (2,), (fn_size-2,))
    center = (f_x_h - f_x)
    return tf.concat((boundary_values[0], center, boundary_values[1]), axis=0)

@tf.function
def hamiltonian(fn, V, epsilon_x):
    """
    Hamiltonian (energy) of the system.

    Returns:
    std, infinitesimal, differental element
    """
    # Free state energy is P^2 / 2m + V(x)
    momentum = P(P(fn)) / tf.cast(2, tf.complex128)
    return V*fn, momentum, 1 / epsilon_x

@tf.function
def P(wave_fn):
    """
    Momentum operator.
    
    Assumes that d/dx --> 0 far from the origin
    """
    bound = tf.zeros(shape=(1,), dtype=tf.complex128)
    return -1j * rolling_diff(wave_fn,
                              boundary_values=(bound, bound))

@tf.function
def td_scrodinger(wave_fn, V, epsilon_x):
    """
    Time-dependent Schrödinger Equation.

    d/dt |psi> = -iH |psi>
    """
    std, infi, diff = hamiltonian(wave_fn, V, epsilon_x)
    return -1j * std, -1j* infi, diff

@tf.function
def _next_state(wave_fn, V, epsilon_x, epsilon_t):
    wave_fn /= tf.norm(wave_fn)
    std, infi, diff =  td_scrodinger(wave_fn, V, epsilon_x)
    update = std * epsilon_t + infi * (epsilon_t * diff)
    return wave_fn + update

@tf.function
def next_state(wave_fn, V, epsilon_x, epsilon_t, n_steps=1):
    epsilon_x = tf.cast(epsilon_x, dtype=tf.complex128)
    epsilon_t = tf.cast(epsilon_t, dtype=tf.complex128)
    for _ in range(n_steps):
        wave_fn = _next_state(wave_fn, V, epsilon_x, epsilon_t)
    return wave_fn

def lump(center, momentum, sigma=1):
    """
    Generator function for a wave lump centered at `center` with momentum `momentum`.
    """
    def fn(x):
        x = tf.cast(x, dtype=tf.complex128)
        return tf.math.exp(-((x-center)/sigma) ** 2 + momentum*1j*x) / tf.cast(np.sqrt(2*np.pi), dtype=tf.complex128)
    return fn

def one_node(center, momentum):
    def fn(x):
        x = tf.cast(x, dtype=tf.complex128)
        return x * lump(center, momentum)(x)
    return fn

def sample_function(func, x_low, x_high, res):
    vec = tf.linspace(x_low, x_high, res)
    return tf.vectorized_map(func, vec)

@tf.function
def expectation_value(obs, state):
    return tf.reduce_sum(tf.math.conj(state)*obs(state))