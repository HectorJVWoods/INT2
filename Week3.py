import numpy as np


def q1a():
    x = [2, 3, 5]
    v = [3, 0, 4]
    xT = np.transpose(x)
    vT = np.transpose(v)
    print(xT * v)
    print(x * vT)



#q1a() # the two are equal.

def q1b():
    A = [[4, 1, 0, 2], [2, 0, 1, 3]]
    b = [3, 0, 5, 1]
    print(np.shape(A))
    print(np.shape(b))
    C = np.matmul(A, b)
    print(C)
#q1b()

def q1c():
    A = [[4, 1, 0, 2], [2, 0, 1, 3]]
    B = [[3, 2, 0], [0, 4, 2], [5, 0, 1], [1 ,3 , 4]]
    C = np.matmul(A,B)
    print(C)

#q1c()

def q1d():
    w = [2, 4, 3, 1]
    w2 = np.linalg.norm(w)
    print(w2)

#q1d()

def unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm # divide vector by its length to get unit vector

def q1e():
    u = [5, 0, 0]
    print(unit_vector(u))

#q1e()


def q2b():
  B = np.array([[3, 2, 0],[0, 4, 2], [5, 0, 1], [1, 3, 4]])
  print(B[1:3, 1])


#q2b()

def dot_product(a, b):
    if(len(a) != len(b)):
        return None
    sum = 0
    for i in range (len(a)):
        sum = sum + (a[i] * b[i])
    return sum


def q2c():
    a = [1,2,3]
    b = [4,5,6]
    print(dot_product(a,b))

#q2c()

#q2d:
tumour_data = [[1, 17, 9], [2, 21, 10], [3, 12, 17]]

def transform(data):
    return np.dot(np.array([1, 1, 1]), data)[1:3]

def q2e():
    print(transform(tumour_data))

#q2e()

def q2f():
    theta = [4,5]
    thetaZero = -100
    thetaT = np.transpose(theta)
    for patient in tumour_data:
        print("---Patient---")
        print(patient)
        x1 = patient[1]
        x2 = patient[2]
        x = [x1,x2]
        result = np.matmul(thetaT,x) + thetaZero
        print(result)


q2f()