import numpy as np

example_data = np.array([[3, 2, 1], [1, 2, 3], [4, 5, 6]])
example_expected = np.array([-1, 1, -1])


def perceptron_with_offset(data, actual, T):
    '''
    Perceptron with an explicit offset theta_0.
    :param data: The data
    :param actual: either 0, 1 or -1; what we expect the datapoint to be classified as
    :param T: A hyperparameter defining how many times we should reiterate before assuming convergance.
    :return: theta and theta_0. they give an equation for a line separating the data of form (thetaT * x) + theta_0 == 0
    '''
    (d, n) = data.shape
    theta = np.zeros(d)
    theta_0 = 0


    for t in range(T):
        for i in range(n):
            yi = actual[i]
            xi = data[i]
            thetaT = np.transpose(theta)
            if yi * (np.matmul(thetaT, xi) + theta_0) <= 0:
                theta = theta + (yi * xi)
                theta_0 = theta_0 + yi

    return theta, theta_0

def perceptron_through_origin(data, actual, T):
    '''
    Perceptron where classifier is guaranteed to be through the origin. Less powerful than with offset but is a bit
    faster.
    :param data: The data we want to classify
    :param actual: either 0, 1 or -1; what we expect the datapoint to be classified as
    :param T: A hyperparameter defining how many times we should reiterate before assuming convergance.
    :return: theta - gives an equation for a line separating the data of form thetaT * x == 0 must go through origin.
    '''
    (n, d) = data.shape
    theta = np.zeros(d+1)
    for t in range(T):
        for i in range(n):
            print(d)
            yi = actual[i]
            xi = data[i]
            thetaT = np.transpose(theta)
            if yi * (np.matmul(thetaT, xi)) <= 0:
                theta = theta + (yi * xi)
    return theta



extra_dimension = np.ones((len(example_data[0]),1))
example_data_append_dimension = np.append(example_data, extra_dimension, axis=1)

print(perceptron_with_offset(example_data, example_expected, 10))
print(perceptron_through_origin(example_data_append_dimension, example_expected, 10))