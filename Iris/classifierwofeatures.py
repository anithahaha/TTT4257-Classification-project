#import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(filename):
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    return data

iris_types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
#organizing data
data = read_data('Iris/iris.data')

#Deletes features
data = np.delete(data, 1, 1) #np.delete(array, obj, axis). axis=0: row, axis=1: column
data = np.delete(data, 0, 1) #np.delete(array, obj, axis). axis=0: row, axis=1: column
data = np.delete(data, 1, 1)

# Shape of data
m, n = data.shape

class_1 = read_data('Iris/class_1')
class_2 = read_data('Iris/class_2')
class_3 = read_data('Iris/class_3')

#data to train 
def seperate_data(nTrain, nTest, placement):
    """Seperates data into train and test
    placement chooses whether training values are first or last n
    NB: use nTest as nTrain param if last n values are training"""
    data_train = np.concatenate((data[0:nTrain], data[50:50+nTrain], data[100:100+nTrain]))
    np.random.shuffle(data_train)
    data_train = data_train.T
    #feature array
    x_train = data_train[0:n-1] 
    x_train = np.concatenate((x_train, np.ones((nTrain*3,1)).T)).T #90, expands feature array with 1 such that W array can include bias w0 
    t_train = data_train[n-1] #labels

    data_test = np.concatenate((data[nTrain:nTrain+nTest], data[50+nTrain:50+nTrain+nTest], data[100+nTrain:100+nTrain+nTest]))
    np.random.shuffle(data_test)
    data_test = data_test.T
    x_test = data_test[0:n-1]
    x_test = np.concatenate((x_test, np.ones((nTest*3,1)).T)).T #60
    t_test = data_test[n-1]
    if placement == 'last':
        x_train, x_test = x_test, x_train
        t_train, t_test = t_test, t_train
    return x_train, x_test, t_train, t_test

def init_params(): 
    #initializes W with zeroes
    W = np.zeros((3, n-1))
    #initializes w0 with ones
    w0 = np.zeros((3, 1))+1
    #combines W and wo [W w0]
    W = np.concatenate((W, w0), axis=1)
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

def ohe(labels):
     #One-hot encoding - gives labels 0 or 1 values based on true class of feature array
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
    #returns gradient of MSE with regards to W
    test = np.multiply(np.multiply(gk-tk, gk), 1-gk)
    grad_MSE = np.matmul(test.reshape((3,1)), xk.T.reshape((1, n)))
    return grad_MSE

def update_W(W, alpha, sum_grad_MSE):
    #updates w
    new_W = W - alpha*sum_grad_MSE
    return new_W

def calcMSE(gk, tk):
    #calculates MSE
    mse_ = 1/2*np.matmul((gk-tk).T,(gk-tk))
    return mse_

def calculate_w(x, t, W, alpha): 
    new_W = W
    grad = 0
    mse = 0
    #calculates gradient of MSE for training set
    for i in range(len(x)):
        zk = z(new_W, x[i])
        gk = g(zk)
        grad += gradient_MSE(gk, t[i], x[i])
        mse += calcMSE(gk, t[i])
    new_W = update_W(new_W, alpha, grad)
    print('MSE', mse)
    #returns updated W
    return new_W

def training(alpha, iter):
    #calculates final W for given alpha and iterations
    W = init_params()
    for j in range(iter):
        W = calculate_w(x_train, t_train, W, alpha)
    print('W', W)
    return W
    
def decision_rule(arr):
    # Find the index of the maximum value
    max_index = np.argmax(arr)
    return max_index

def one_hot_max(arr):
    max_index = decision_rule(arr)
    # Create a one-hot encoded array with a 1 at the maximum index
    one_hot = np.zeros_like(arr)
    one_hot[max_index] = 1
    return one_hot

def histogram(class1, class2, class3):
    bins = np.linspace(0, 8, 100)
    features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    for i in range(len(class1[0])):
        plt.hist(class1[:,i], bins, alpha=0.5, label='Class 1: Iris setosa')
        plt.hist(class2[:,i], bins, alpha=0.5, label='Class 2: Iris versicolor')
        plt.hist(class3[:,i], bins, alpha=0.5, label='Class 3: Iris virginica')
        plt.legend(loc='upper right')
        plt.ylabel('Count')
        plt.title(features[i] + '[cm]')
        plt.plot()
        plt.show()

def confusion_matrix(expected_labels, predicted_labels, iris_types):
    num_classes = 3
    iris_types = np.array(iris_types)

    # Initialize matrix with zeros
    conf_mat = np.zeros((num_classes, num_classes))
    
    # Fill values
    for i in range(len(expected_labels)):
        conf_mat[np.where(iris_types == expected_labels[i]), np.where(iris_types == predicted_labels[i])] += 1

    # Print matrix
    print("\nConfusion matrix: ")
    print(conf_mat)

    return conf_mat

def expected_predicted_val(iris_types, data, label):
    error = 0
    correct = 0 
    total = 0

    expected_values = []
    predicted_values = []

    for i in range(len(label)):
        t = np.array(label[i])
        expected_values.append(iris_types[np.argmax(t)])

        gk = (one_hot_max(np.abs(g(z(W_final.reshape((3,n)), data[i].T.reshape((n, 1)))))).T)[0]
        predicted_values.append(iris_types[np.argmax(gk)])
        
        if ((t == gk ).all()):
            correct += 1
        
        else:
            error += 1
        total += 1
        
    print("Error: ", error,"(",round(error/total,2)*100 ,"%)")
    print("Correct: ", correct, "(",round(correct/total,2)*100, "%)")
    print("Total: ", total)  
     
    return expected_values, predicted_values, error, correct, total
print(data.shape)

x_train, x_test, t_train, t_test = seperate_data(30, 20, 'first')
t_test = ohe(t_test)
t_train = ohe(t_train)

W_final = training(alpha=0.01, iter=4000) #likte tall: 0.00045,0.002,0.003, 20000
expected_values, predicted_values, error, correct, total = expected_predicted_val(iris_types, x_test, t_test)
confusion_matrix(expected_values, predicted_values, iris_types)

#histogram(class_1, class_2, class_3)