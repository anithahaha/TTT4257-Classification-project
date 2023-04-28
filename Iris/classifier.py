#import pandas as pd
import csv
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#rom sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

def read_data(filename):
    data = pd.read_csv(filename)
    np.array(data)
    return data

#organizing data
data = read_data('koding/iris.data')
m, n = data.shape

class_1 = read_data('koding/class_1')
class_2 = read_data('koding/class_2')
class_3 = read_data('koding/class_3')

#data to train 
data_train = np.concatenate((data[0:30], data[50:80], data[100:130]))
np.random.shuffle(data_train)
data_train = data_train.T
x_train = data_train[0:n-1]
x_train = np.concatenate((x_train, np.ones((90,1)).T)).T #90

#må legge til array [1] for bias
# labels til k (k er index for hver blomst)
t_train = data_train[n-1]

#data to test
data_test = np.concatenate((data[30:50], data[80:100], data[130:150]))
np.random.shuffle(data_test)
data_test = data_test.T
x_test = data_test[0:n-1]
x_test = np.concatenate((x_test, np.ones((59,1)).T)).T #60, why only 59?

t_test = data_test[n-1]

#print(x_train[:, 0]) x1, first column

def init_params(): #velges tilfeldig
    W = np.random.randn(3, 4)
    w0 = np.random.randn(3, 1)

    # W.shape = (3,5)
    W = np.concatenate((W,w0), axis=1)
    return W

def z(W, xk):
    zk = np.matmul(W, xk)
    return zk

def sigmoid(zik):
    return 1/(1 + np.exp(-zik))

def g(zk):
    sigmoide = np.vectorize(sigmoid)
    gk = sigmoide(zk)
    return gk

def ohe(labels): #gives labels 0 or 1 values
    iris_types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    mapping = {}
    for x in range(len(iris_types)):
        mapping[iris_types[x]] = x
    ohe_T = []

    for c in labels:
        arr = list(np.zeros(len(iris_types), dtype = int))
        arr[mapping[c]] = 1
        ohe_T.append(arr)
    return ohe_T

def gradient_MSE(gk, tk, xk):
    #print("xk", xk.shape)
    #print("gk", gk.shape)
    #print("tk", tk.shape)
    #print("xk.T", (xk.T).shape)
    test = np.multiply(np.multiply(gk-tk, gk), 1-gk)
    grad_MSE = np.matmul(test.reshape((3,1)), xk.T.reshape((1, 5)))
    #print("grad_MSE", grad_MSE.shape)

    return grad_MSE

def update_W(W, alpha, sum_grad_MSE): #kan være feil
    new_W = W - alpha*sum_grad_MSE
    return new_W

def chunks(array, L):
    array = np.array_split(array, L)
    return array

def calculate_chunks(xChunk, tChunk, W, alpha): #calculates gradient of MSE for each chunk, returns final W
    grad = 0
    for i in range(len(xChunk)):
        zk = z(W, xChunk[i])
        gk = g(zk)
        grad += gradient_MSE(gk, tChunk[i], xChunk[i])
    new_W = update_W(W, alpha, grad)
    #print("new W")
    #print(new_W)
    return new_W

def training(alpha, L):
    W = init_params()
    x = chunks(x_train, L)
    t = chunks(Tn, L)
    for i in range(len(x)):
        W = calculate_chunks(x[i], t[i], W, alpha)
    return W
    
def decision_rule():
    return;

Tn = ohe(t_train)

W_final = training(alpha=0.4, L=5)
t_test = ohe(t_test)

for i in range(50):
    print("\nTest", i)
    print("t:", t_test[i])
    #print(t_test[i])
    print("gk:")
    print(z(W_final.reshape((3,5)), x_test[i].T.reshape((5, 1))))

    z = z(W_final.reshape((3,5)), x_test[i].T.reshape((5, 1)))
    print((z[:,i]))

    for k in range(z.size):
        if (z[:,i].max() == z[k,i]):
            z[k] = 1
        else:
            z[k] = 0


