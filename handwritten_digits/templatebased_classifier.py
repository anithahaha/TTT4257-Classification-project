# Imports
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
import math
from scipy.spatial import distance

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

#print('xtrain', x_train.shape)

x_test = read_data("handwritten_digits/test_images.bin", 'image', 10000)
y_test = read_data("handwritten_digits/test_labels.bin", 'label', 10000)

# Calculates euclidean distance
def euclidean_distance(img1, img2):
    return math.dist(img1,img2)

# Calculates k nearest neighbours
def kNN(k, x, x_train, y_train):
    distance = []

    # Calculate distance between each training value
    for ref in x_train:
        distance.append(euclidean_distance(x, ref))


    # Calculate index value of k NN
    k_index = np.argsort(distance)[:k]

    k_label = [y_train[i] for i in k_index]

    # Perform majority vote
    count = Counter(k_label)
    pred = count.most_common(1)[0][0]

    return pred

def test(k, x_test, x_train, y_train, y_test):
    prediction = []
    actual = y_test 
    num_x = 5
    for i in range(num_x): 
        pred_value = kNN(k, x_test[i], x_train, y_train)
        prediction.append(pred_value)
    confusion_matrix(prediction, actual[:num_x])
    error_rate(prediction, actual[:num_x])
    return prediction, actual[:num_x]

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

    print("Error: ", error, "\nCorrect: ", correct, "\nError rate: ", round((error/num_elems)*100,2), "%")
    
    return correct, error, num_elems

# Calculates and displays confusion matrix
def confusion_matrix(expected_values, predicted_values):
    num_classes = 10

    # Initialize matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    # Fill values
    for i in range(len(expected_values)):
        conf_matrix[expected_values[i]][predicted_values[i]] += 1

    print("Confusion matrix\n", conf_matrix.astype(int))

    return conf_matrix

# Displays image (both classified and misclassified)
def display_image(image_array):
    plt.imshow(image_array.reshape(28,28), cmap='gray')
    plt.show()

# Sort values by class [0,9]
def sort_values(values, labels):
    
    sorted_labels = np.argsort(labels)
    sorted_values = [values[i] for i in sorted_labels]
    sorted_values = np.array(chunks(sorted_values, 60000)).reshape(60000, 784)
    count = Counter(labels)

    return sorted_values, count

def cluster(values, labels):
    sorted_values, count = sort_values(values, labels)
    num_clusters = 64

    # Calculating indices
    starts = 0
    ends = count[0]

    # Array with all cluster_centers.
    cluster_labels = []
    cluster_values = []

    # Every class
    for i in range(10):
        # 1D array for the centers in each cluster
        values_one_class = sorted_values[starts:ends]

        starts = ends
        ends = starts + count[i]

        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(values_one_class)
        
        # Save values for class in array
        
        cluster_labels_class = [i for k in range(num_clusters)]
        cluster_labels.append(cluster_labels_class)

        cluster_values.append(kmeans.cluster_centers_)

    cluster_labels = np.array(cluster_labels).flatten()
    cluster_values = np.array(chunks((np.array(cluster_values)).flatten(), 640)).reshape(640, 784)

    return cluster_labels, cluster_values 


# Test for assignment part 2 (KNN with clustering)
def test_clustering(x_test, y_test, x_train, y_train, k=1):
    cluster_labels, cluster_values = cluster(x_train, y_train)
    prediction, expectation = test(k, x_test, cluster_values, cluster_labels, y_test)
    prediction, expectation = np.array(prediction), np.array(expectation)

    confusion_matrix(expectation, prediction)
    error_rate(expectation, prediction)

    return prediction, expectation


# Test for assignment part 1 (KNN without clustering, first for k = 1 then for k = 7)
print("Starting classifying k = 1")
test(1, x_test, x_train, y_train, y_test)

print("Starting classifying k = 7")
test(7, x_test, x_train, y_train, y_test)

# Test clustering
print("Starting classifying with clustering")
test_clustering(x_test, y_test, x_train, y_train)
