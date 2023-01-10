import random

from ldl import c01


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

# Perceptron training loop.
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list)  # Randomize the order
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = c01.compute_output(w, x)  # Perceptron function

        if y != p_out:  # If "wrong," update weights
            for j in range(0, len(w)):
                w[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(w)  # Show update weights
