import tensorflow as tf
import numpy as np

from backend import sample_function, lump, expectation_value, next_state, P


def spring_potential(x):
    """ Potential for harmonic oscilator """
    return tf.cast(x ** 2, dtype=tf.complex128)


def barrier_potential_1(x):
    """ A sample potential that should produce a tunneling effect """
    return tf.cast(5 * tf.sin(x) / (1 + x ** 2), dtype=tf.complex128)


def barrier_potential_2(x):
    """ Another sample potential that should produce a tunneling effect """
    return tf.cast(1 / (1 + x ** 2) + (x / 4) ** 2, dtype=tf.complex128)


x_low = -15
x_high = 15
n_pts = 2000

# the small change in x between neighboring places in the array
epsilon_x = (x_high - x_low) / n_pts


def main(total_steps, steps_per_call):

    wf = sample_function(lump(2, 0, sigma=2), x_low, x_high, n_pts)
    wf /= tf.linalg.norm(wf)
    potential = sample_function(barrier_potential_2, x_low, x_high, n_pts)

    data = [wf]

    for i in range(total_steps // steps_per_call):
        if i % 100 == 0:
            print(f"Loop: {i}", end="\r")
        wf = next_state(
            wf,
            epsilon_x=epsilon_x,
            epsilon_t=0.0001,
            V=potential,
            n_steps=steps_per_call,
        )
        data.append(wf)

    print()
    return data


if __name__ == "__main__":
    data = main(1000000, 100)
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(nrows=2)

    mag, p = ax

    mag.imshow(
        [v / np.linalg.norm(v) for v in [np.absolute(wf.numpy()) for wf in data]],
        interpolation="nearest",
        aspect="auto",
    )
    p.plot(
        [
            np.real(_p.numpy())
            for _p in tf.vectorized_map(
                tf.function(lambda x: expectation_value(P, x)), tf.stack(data)
            )
            / (x_high - x_low)
        ]
    )
    plt.show()