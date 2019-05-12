import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def inicial_var(X , Y):
    m = int(Y.shape[1])
    n_x= int(X.shape[0])
    n_h= 4
    n_y= int (Y.shape[0])
    return n_x,n_h,n_y

def inicial_par(nx,nh,ny):
    np.random.seed(2)
    w1 = np.random.rand(nh,nx)*0.01
    b1 = np.zeros((nh,1))
    w2 = np.random.rand(ny,nh)*0.01
    b2 = np.zeros((1,ny))
    return w1,b1,w2,b2

def PropagateFor (X_ , w_1 , w_2 , b_1 , b_2):
    z1 = np.dot(w_1,X_) + b_1
    a1 = np.tanh(z1)
    z2 =  np.dot(w_2,a1) + b_2
    a2 = sigmoid(z2)
    return a1 , a2

def propagateback (parameters,X1,Y1,cache):
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    m = int(Y1.shape[1])
    Dz2 = A2 - Y1
    Dw2 = np.dot(Dz2,A1.T)/m
    Db2 = np.sum(Dz2,axis = 1 ,keepdims = True)/m
    Dz1 = np.dot(W2.T,Dz2) * (1- (A1**2))
    Dw1 = np.dot(Dz1,X1.T)/m
    db = np.sum(Dz1,axis = 1,keepdims = True)/m
    return Dw2 , Db2 , Dw1 , db


def update_parameters(parameters, grads, learning_rate=1.2):
    w1 = parameters['W1']
    w2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    dw2 = grads['dW2']
    dw1 = grads['dW1']
    db1 = grads['db1']
    db2 = grads['db2']
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
    parameters = {'W1':w1 ,'W2': w2 , 'b1':b1 ,'b2':b2}
    return parameters

def cost_fun (A_1 , Y_):
    m = int(Y.shape[1])
    cost =(np.dot(np.log(A_1),Y_.T) +  np.dot(np.log(1 - A_1), 1-Y_.T) ) / -m
    return cost


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x , hh ,n_y = inicial_var(X , Y)
    W1,b1,W2,b2 = inicial_par (n_x , n_h , n_y)
    pares = {'W1': W1 , 'W2' : W2 , 'b1' : b1 , 'b2' : b2}
    for i in range(num_iterations):
        a1 , a2 = PropagateFor(X,pares['W1'],pares['W2'],pares['b1'],pares['b2'])
        cost = cost_fun(a2 , Y)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        DW2 , Db2 , DW1 , Db1 = propagateback (pares , X,Y , {'A1' : a1 , 'A2' : a2 } )
        pares = update_parameters(pares ,{'dW2' : DW2 ,'dW1': DW1 ,'db1' : Db1 , 'db2':Db2 })
    return pares

def predict(pares, X) :
    a1 , a2 = PropagateFor(X,pares['W1'],pares['W2'],pares['b1'],pares['b2'])
    return np.round(a2)

'''
X_assess, Y_assess = nn_model_test_case()
W1 = np.array([[-0.00416758,-0.00056267],[-0.02136196,0.01640271],[-0.01793436,-0.00841747], [ 0.00502881 ,-0.01245288]])
b1  = np.array([[ 0.],[ 0.],[ 0.],[ 0.]])
W2  = np.array([[-0.01057952,-0.00909008,0.00551454,0.02292208]])
b2  = np.array([[ 0.]])
a1 , a2 = PropagateFor (X_assess , W1 , W2 , b1 , b2)
print a2
pares = {'W1': W1 , 'W2' : W2 , 'b1' : b1 , 'b2' : b2}
DW2 , Db2 , DW1 , Db1 = propagateback (pares , X_assess,Y_assess, {'A1' : a1 , 'A2' : a2 } )
pares = update_parameters(pares ,{'dW2' : DW2 ,'dW1': DW1 ,'db1' : Db1 , 'db2':Db2 })
print pares
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
print parameters
'''
'''


# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
'''
X, Y = load_planar_dataset()
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
