import random

import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x is None:
        return None
    if x < 0:
        return -1
    if x == 0:
        return 0
    return 1

def linear_classify(x, theta, theta_0):
    """Uses the given theta, theta_0, to linearly classify the given data x. This is our hypothesis or hypothesis class.

    :param x: a vector - the current datapoint
    :param theta: a vector - theta for the current hypothesis
    :param theta_0: theta 0 for the current hypothesis
    :return: 1 if the given x is classified as positive, -1 if it is negative, and 0 if it lies on the hyperplane.
    """
    thetaT = np.transpose(theta)
    return sign(np.matmul(theta, x) + theta_0)


def Loss(prediction, actual):
    """Computes the loss between the given prediction and actual values. I will use Mean-Squared error.

    :param prediction: the predicted value by the classifier
    :param actual: the actual value for this datapoint
    :return: Mean squared error between prediction and actual
    """
    if(len(prediction) != len(actual)):
        raise "Prediction has a different length to actual!"
    n = len(prediction)
    return (1/n) * np.sum(((prediction - actual)**2))

print(Loss(np.array([1,2]), np.array([3,4])))

def E_n(h, data, labels, L, theta, theta_0):
    """Computes the error for the given data using the given hypothesis and loss.

    :param h: Hypothesis class, for example a linear classifier.
    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param L: A loss function to compute the error between the prediction and label.
    :param theta:
    :param theta_0:
    :return:
    """
    (d, n) = data.shape
    num_incorrect = 0
    for i in range(n):
        data_point = data[i]
        predicted = h(data_point, theta, theta_0)
        actual = labels[i]
        loss = L(predicted, actual)
        if loss != 0: # since our results are either 0, -1, or 1, any deviation suggests the model is wrong.
            num_incorrect = num_incorrect + 1
    return n / num_incorrect # proportion of total that were wrong

def random_linear_classifier(data, labels, params={}, hook=None):
    """
    Works by choosing k random theta and theta_0 (where k is a hyperparameter roughly analogous to 'temperature')
    and chooses the parameters that result in the lowest possible loss.
    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return:
    """
    k = params.get('k', 100)  # if k is not in params, default to 100
    (d, n) = data.shape

    theta = np.zeros((n, k))
    theta_0 = np.array((n, k))
    errors = np.array((n, k))


    for j in range(k):
        for i in range(d):
            theta[j][d] = random.randint(0, d)
        theta_0[j] = random.randint(0,d*1000) # should really be any real number but i can't be bothered
        errors[j] = E_n(linear_classify, data, labels, Loss, theta[j], theta_0[j])


    j_star = np.argmin(errors)
    return theta[j_star], theta_0[j_star]


def perceptron(data, labels, params={}, hook=None):
    """The Perceptron learning algorithm.

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return:
    """


    T = params.get('T', 100)  # if T is not in params, default to 100
    (d, n) = data.shape

    # Todo: Implement the Perceptron algorithm here.
    pass


def plot_separator(plot_axes, theta, theta_0):
    """Plots the linear separator defined by theta, theta_0, into the given plot_axes.

    :param plot_axes: Matplotlib Axes object
    :param theta:
    :param theta_0:
    """

    # One way we can plot the intercept is to compute the parametric line equation from the implicit form.
    # compute the y-intercept by setting x1 = 0 and then solving for x2:
    y_intercept = -theta_0 / theta[1]
    # compute the slope (-theta[0]/theta[1], I think)
    slope = -theta[0] / theta[1]
    # Then compute two points using:
    xmin, xmax = -15, 15
    # Note: It's not ideal to only plot the lines in a fixed region, but it makes this code simple for now.

    p1_y = slope * xmin + y_intercept
    p2_y = slope * xmax + y_intercept

    # Plot the separator:
    plot_axes.plot([xmin, xmax], [p1_y.flatten(), p2_y.flatten()], '-')
    # Plot the normal:
    # Note: The normal might not appear perpendicular on the plot if ax.axis('equal') is not set - but it is
    # perpendicular. Resize the plot window to equal axes to verify.
    plot_axes.arrow((xmin + xmax) / 2, (p1_y.flatten() + p2_y.flatten()) / 2, float(theta[0]), float(theta[1]))


if __name__ == '__main__':
    """
    We'll define data X with its labels y, plot the data, and then run either the random_linear_classifier or the
    perceptron learning algorithm, to find a hypothesis h from the class of linear classifiers.
    We then plot the best hypothesis, as well as compute the training error. 
    """

    # Let's create some training data and labels:
    #   X is a d x n matrix where d is the number of data dimensions and n the number of examples. So each data point
    #     is a column vector.
    #   y is a 1 x n matrix with the label (actual value) for each data point.
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])

    # To test your algorithm on a larger dataset, uncomment the following code. It generates uniformly distributed
    # random data in 2D, along with their labels.
    # X = np.random.uniform(low=-5, high=5, size=(2, 20))  # d=2, n=20
    # y = np.sign(np.dot(np.transpose([[3], [4]]), X) + 6)  # theta=[3, 4], theta_0=6

    # Plot positive data green, negative data red:
    colors = np.choose(y > 0, np.transpose(np.array(['r', 'g']))).flatten()
    plt.ion()  # enable matplotlib interactive mode
    fig, ax = plt.subplots()  # create an empty plot and retrieve the 'ax' handle
    ax.scatter(X[0, :], X[1, :], c=colors, marker='o')
    # Set up a pretty 2D plot:
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Linear classification")


    # We'll define a hook function that we'll use to plot the separator at each step of the learning algorithm:
    def hook(params):
        (th, th0) = params
        plot_separator(ax, th, th0)


    # Run the RLC or Perceptron: (uncomment the following lines to call the learning algorithms)
    theta, theta_0 = random_linear_classifier(X, y, {"k": 100}, hook=None)
    # theta, theta_0 = perceptron(X, y, {"T": 100}, hook=None)
    # Plot the returned separator:
    # plot_separator(ax, theta, theta_0)

    # Run the RLC, plot E_n over various k:
    # Todo: Your code

    print("Finished.")