import numpy as np

example_data = np.array([[3, 2, 1], [1, 2, 3], [4, 5, 6]])
example_expected = np.array([-1, 1, -1])


def perceptron(data, actual, T):
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


example_trained_model = perceptron(example_data, example_expected, 10)
print(example_trained_model)