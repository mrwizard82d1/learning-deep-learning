import random

import matplotlib.pyplot as plt


# Define variables for plotting.
color_list = ['r-', 'm-', 'y-', 'c-', 'b-', 'g-']
color_index = 0


def show_learning(weights):
    """Show learning"""
    global color_index

    print(f'weights0={weights[0]:5.2f}, weights1={weights[1]:5.2f}, weights2={weights[2]:5.2f}')

    if color_index == 0:
        plt.plot([1.0], [1.0], 'b_', markersize=12)
        plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], 'r+', markersize=12)
        plt.axis([-2, 2, -2, 2])
        plt.xlabel('x1')
        plt.xlabel('x2')

    x_ = [-2.0, 2.0]
    if abs(weights[2]) < 1e-5:
        y_ = [-weights[1] / 1e-5 * -2.0 + (-weights[0] / 1e-5),
              -weights[1] / 1e-5 * 2.0 + (-weights[0] / 1e-5)]
    else:
        y_ = [-weights[1] / weights[2] * -2.0 + (-weights[0] / weights[2]),
              -weights[1] / weights[2] * 2.0 + (-weights[0] / weights[2])]
    plt.plot(x_, y_, color_list[color_index])
    if color_index < (len(color_list) - 1):
        color_index += 1


# Define the variables needed to control the training process.
random.seed(7)  # Supply a seed to make runs repeatable
LEARNING_RATE = 0.1  # A constant learning rate
index_list = [0, 1, 2, 3]  # Used to randomize the order


# Define the training examples.
# Note that each input example begins with x[0], the bias term, which is always 1.
x_train = [(1, -1, -1),
           (1, -1, 1),
           (1, 1, -1),
           (1, 1, 1)]  # inputs
y_train = [1, 1, 1, -1]  # Expected output (ground truth)

# Define the perceptron weights
w = [0.2, -0.6, 0.25]  # initialize to some "random" values

# Print initial weights
show_learning(w)


def compute_output(w_, x_):
    """
    Compute the output of the perceptron with weights, `w_` and vector, `x_`.

    Note that the length of the vectors, `w_` and `x_` must both be identically equal to `n + 1` where `n` is the
    number of actual inputs (excluding the "special", $x_{0}$, component).

    Args:
        w_: The weights for the calculation.
        x_: The vector whose output is to be calculated. (Note that `$x_{0}$` is identically 1.)

    Returns:
        The computed output of the perceptron.
    """

    z = sum([x_[k] * w_[k] for k in range(len(w_))])
    return -1 if z < 0 else 1


# Perceptron training loop.
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)  # Randomize the order
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x)  # Perceptron function

        if y != p_out:  # If "wrong," update weights
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(w)  # Show update weights

plt.show()
