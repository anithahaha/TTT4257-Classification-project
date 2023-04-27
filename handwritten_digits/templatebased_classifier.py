# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def chunks(array, L):
    array = np.array_split(array, L)
    return array

def read_data(filename, type, size):
    with open(filename, 'r') as fid:
        #sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint8)
        #data = data.reshape((-1, channels))
    if type == 'image':
        data = data[16:]
        data = chunks(data, size)
        data = np.array(data)
        
    if type == 'label':
        data = data[8:]
    return data

x_train = read_data("handwritten_digits/train_images.bin", 'image', 60000)
y_train = read_data("handwritten_digits/train_labels.bin", 'label', 60000)

x_test = read_data("handwritten_digits/test_images.bin", 'image', 10000)
y_test = read_data("handwritten_digits/test_labels.bin", 'label', 10000)

# Calculates euclidean distance
def euclidean_distance(q, p):
    return np.linalg.norm(q-p)

# Calculates k nearest neighbours
def kNN(k, x, x_train, y_train):
    distance = []

    #calculate distance between each training value
    for ref in x_train:
        distance.append(euclidean_distance(x, ref))

    #calculate index value of k NN
    k_index = np.argsort(distance)[:k]
    k_label = [y_train[i] for i in k_index]
    #print(k_label)
    #perform majority vote
    count = Counter(k_label)
    pred = count.most_common(1)[0][0]
    return pred

def test(k, x_test, x_train, y_train, y_test):
    prediction = []
    actual = y_test #egt ikke nødvendig
    for i in range(1000):
        pred_value = kNN(k, x_test[i], x_train, y_train)
        prediction.append(pred_value)
        print(f'Counter: {i}')
        print(f'Predicted: {pred_value}, Actual: {actual[i]}\n')
        #image = x_test[i].reshape(28,28)
        # Display image
        #plt.imshow(image, cmap='gray')
        #plt.show()
    return prediction, actual

# Calculates error rate given expected values and predicted values
def error_rate(expected_values, predicted_values):
    correct = 0
    error = 0

    # Checks each testdata if the classification matches the label
    for i in range(len(expected_values)):
        if (expected_values[i] == predicted_values[i]):
            correct += 1
        else:
            error += 1
    num_elems = len(expected_values)

    return correct, error, num_elems

# Calculates and displays confusion matrix
def confusion_matrix(expected_values, predicted_values, num_classes):
    
    # Initialize matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    # Fill values
    for i in range(len(expected_values)):
        conf_matrix[expected_values[i]][predicted_values[i]] += 1

    return conf_matrix

# Displays image (both classified and misclassified)
def display_image(image_array):
    plt.imshow(image_array, cmap='gray')
    plt.show()
    return;

# 
def clustering(data):
    kmeans = KMeans(64)
    kmeans.fit(data)
    print(kmeans.labels_)
    #plt.scatter(x, y, c=kmeans.labels_)
    return;

def retrieve_info(cluster_labels,y_train, kmeans):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label"""
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
    num = np.bincount(y_train[index==1]).argmax()
    reference_labels[i] = num
    return reference_labels

clustering(x_train) 

""" print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)
 """
prediction, actual = test(1, x_test, x_train, y_train, y_test)
print(confusion_matrix(prediction, actual, 10))


#for i in range(len(prediction)):
#    print(f'Predicted: {prediction[i]}, Actual: {actual[i]}\n')

