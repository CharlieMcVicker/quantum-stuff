import tensorflow as tf
import numpy as np

from backend import sample_function, lump, expectation_value, next_state, P

def spring_potential(x):
    return tf.cast(x ** 2, dtype=tf.complex128)

def barrier_potential_1(x):
    return tf.cast(5*tf.sin(x) / (1 + x ** 2), dtype=tf.complex128)

def barrier_potential_2(x):
    return tf.cast(1/(1+x**2) + (x/4)**2, dtype=tf.complex128)
    
x_low = -15
x_high = 15
n_pts = 1000

wf = sample_function(lump(2, 0, sigma=2), x_low, x_high, n_pts)
wf /= tf.linalg.norm(wf)
potential = sample_function(barrier_potential_2, x_low, x_high, n_pts)

epsilon_x = (x_high-x_low)/n_pts

data = [wf]

avg_p = [expectation_value(P, wf) / (x_high - x_low)]

for i in range(1000):
    if i % 100 == 0:
        print(f'Loop: {i}', end='\r')
    wf = next_state(wf, epsilon_x=epsilon_x, epsilon_t=0.0001, V=potential, n_steps=100)
    data.append(wf)
    avg_p.append(expectation_value(P, wf) / (x_high - x_low))

print()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2)

mag, p = ax

mag.imshow([v / np.linalg.norm(v) for v in [np.absolute(wf.numpy()) for wf in data]], interpolation="nearest", aspect="auto")
p.plot([np.real(_p.numpy()) for _p in avg_p])
plt.show()