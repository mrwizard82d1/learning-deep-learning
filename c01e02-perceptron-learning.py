import random


def show_learning(w):
    """Print the current weights"""
    print(f'w0={w[0]:5.2f}, w1={w[1]:5.2f}, w2={w[2]:5.2f}')


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


def compute_output(w, x):
    """
    Compute the output of the perceptron with weights, `w` and vector, `x`.

    Note that the length of the vectors, `w` and `x` must both be identically equal to `n + 1` where `n` is the
    number of actual inputs (excluding the "special", $x_{0}$, component).

    Args:
        w: The weights for the calculation.
        x: The vector whose output is to be calculated. (Note that `$x_{0}$` is identically 1.)

    Returns:
        The computed output of the perceptron.
    """

    z = sum([x[i] * w[i] for i in range(len(w))])
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
