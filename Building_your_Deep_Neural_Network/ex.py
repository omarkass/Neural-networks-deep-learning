import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    w1 = np.random.rand(n_h,n_x) * 0.01
    b1 = np.zeros([n_h,1])
    w2 = np.random.rand(n_y,n_h) * 0.01
    b2 = np.zeros([n_y,1])
    parameters = {'W1':w1,'W2':w2 , 'b1' : b1 , 'b2' : b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    parameters = {}
    np.random.seed(3)
    l = len(layer_dims)
    for i in range( 1 , l ):
        parameters ['W' + str(i)] = np.random.rand( layer_dims[i] , layer_dims[i-1]) * 0.01
        parameters ['b' + str(i)] = np.zeros ([layer_dims[i] , 1])
    return parameters

def linear_forward(A, W, b):
   return np.dot(W,A) + b


def linear_activation_forward(A_prev, W, b, activation):
    z = linear_forward(A_prev,W,b)
    if (activation == 'sigmoid'):
        a,z = sigmoid(z)
    else :
        a,z = relu(z)
    return a

def L_model_forward(X, parameters):
    a = X
    AL = {}
    for i in range(len(parameters)/2):
        if i == (len(parameters)/2) - 1 :
            act = "sigmoid"
        else:
            act = "relu"
        a = linear_activation_forward(a , parameters['W' + str(i+1)] , parameters['b' +str(i+1)],act)
        AL[i+1] = a
    return a , AL

def cost (A,Y):
    m =  int((A.shape)[1])
    return  (np.dot(np.log(A),Y.T) -  np.dot( np.log(1 - A), 1 - Y.T)) /- m

def linear_backward(dZ, cache):
    m = int(dZ.shape[1])
    da_prev =  np.dot (cache[1].T,dZ)
    dw = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ) / m
    return da_prev , dw , db

def linear_activation_backward(dA, cachee, activation):
    if activation == "relu" :
        dZ =  relu_backward(dA,cachee[1])
    else:
        dZ = sigmoid_backward(dA,cachee[1])
    dA_prev, dW, db = linear_backward(dZ,cachee[0])
    return dA_prev, dW, db

def L_model_backward(AL, Y_assess, caches):
    da =  (-Y_assess/AL) + (1-Y_assess)/(1-AL)
    grad = {}
    for i in range(len(caches) , 0 , -1 ):
        if (i == len(caches)) :
            act = "sigmoid"
        else :
            act = "relu"
        da , dw , db = linear_activation_backward(da,caches[i-1],act)
        grad["dA" + str(i)] = da
        grad["dW" + str(i)] = dw
        grad["db" + str(i)] = db

    return grad

def update_parameters(parameters, grads, up_rate):
    para = {}
    for i in range(len(parameters)/2):
        para['W'+str(i+1)] = parameters['W'+str(i+1)] - (up_rate)*grads['dW'+str(i+1)]
        para['b'+str(i+1)] = parameters['b'+str(i+1)] - (up_rate)*grads['db'+str(i+1)]

    return para

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = " + str(parameters["W1"]))
print ("b1 = " + str(parameters["b1"]))
print ("W2 = " + str(parameters["W2"]))
print ("b2 = " + str(parameters["b2"]))
print ("W3 = " + str(parameters["W3"]))
print ("b3 = " + str(parameters["b3"]))
