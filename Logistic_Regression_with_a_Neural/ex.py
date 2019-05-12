from PIL import Image
import PIL
import numpy as np
import glob
import h5py
import matplotlib.pyplot as plt
imgs = np.array(glob.glob("*.jpg"))
pixels = np.empty([int(imgs.shape[0]) , 64, 64 ,3])
train_labels = np.zeros( [int(imgs.shape[0])] )
m_test = 0
m_train = 0
weight = 0
def ChangeSize():
    for n in range(int(imgs.shape[0])):
            x = imgs[n]
            img = Image.open(x)
            img = img.resize((64,64),Image.ANTIALIAS)
            pixels[n] = img
            img.save(x,optimize=True,quality=95)
def GetData ():
    filename = "train_catvnoncat.h5"
    filename2 = "test_catvnoncat.h5"
    f = h5py.File(filename , 'r')
    f2 = h5py.File(filename2 , 'r')
    return list(f['train_set_x']) , list(f['train_set_y']) , list(f2['test_set_x']) , list(f2['test_set_y'])

def prepare (x_train , y_train  , x_test , y_test):
    global weight
    global m_train
    global m_test
    weight = len(x_train[0])
    m_train = len(x_train)
    m_test = len(x_test)
    X_test = np.reshape(x_test,(weight*weight*3 , -1 )) / 225.
    X_train = np.reshape(x_train,(weight*weight*3 , -1 )) / 225.
    Y_train = np.reshape(y_train , (1,-1))
    Y_test = np.reshape(y_test , (1,-1))
    return X_train , Y_train , X_test , Y_test

def SetVariables():
    for x in range(int(imgs.shape[0])):
        pixels[x] = Image.open(imgs[x])
        str = "cat"
        if imgs[x].find(str) == 0:
            train_labels[x] = 1
    m = int(imgs.shape[0])
    weight = int(pixels.shape[1])

def sigmoid (x):
        return (1/(np.exp(-x) + 1))

def parameters(Npixels):
    W = np.zeros([Npixels,1])
    B = 0
    return W , B

def propogate(w,b,x,y):
    m = int(y.shape[0])
    a = sigmoid(np.dot(w.T,x) + b)
    cost =  np.sum(y*np.log(a) + (1-y)*np.log(1-a) ,axis = 1)/-m
    db =(np.sum( a-y,axis = 1 ))/m
    dw = (np.dot ( x, (a-y).T ))/m
    return cost , dw , db

def optmizaition (w1,b1,x1,y1,n_interacition , learning_rate):
    costs = []
    for x in range (4):
            print x1.shape
            print ( "is the "+ str(x) + str(w1.shape) + '/n')
            cost , dw , db = propogate(w1,b1,x1,y1)
            print ( "is the "+ str(x) + str(w1.shape))
            w1 = w1 - learning_rate * dw
            b1 = b1 - learning_rate * db
            if x % 100 == 0:
                costs.append(cost)
    pares = {"w":w1,"b":b1 }
    gradient = {"db":db ,"dw":dw }
    return costs,pares,gradient

def predict (x , W , B):
    pre = sigmoid (np.dot(W.T , x) + B )
    pre = np.where (pre > 50 , 0 , 1)
    return pre

def model1(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w = np.zeros((weight*weight*3,1))
    b = 0
    costs,parameters, grads = optmizaition(w, b, X_train, Y_train, num_iterations, learning_rate)
    print (costs )


train_x , train_y , test_x , test_y = GetData ()
xtrain , ytrain ,xtest , ytest   = prepare(train_x , train_y ,test_x , test_y)
model1(xtrain,ytrain,xtest,ytest)
