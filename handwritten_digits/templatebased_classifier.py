# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
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

print('xtrain', x_train.shape)

x_test = read_data("handwritten_digits/test_images.bin", 'image', 10000)
y_test = read_data("handwritten_digits/test_labels.bin", 'label', 10000)

# Calculates euclidean distance
def euclidean_distance(img1, img2):
    #np.linalg.norm(q-p)
    #math.dist(q, p)
    a = img1-img2 # Since we import uint8, this will give zero for img1[i]-img2[i] < 0
    b = np.uint8(img1<img2) * 254 + 1 # smart trick
    return np.sum(a*b) 

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
    for i in range(20): #change
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

# Cluster data into 64 clusters
""" def clustering(data):
    kmeans = KMeans(64)
    kmeans.fit(data)
    print(kmeans.labels_)
"""  """    #plt.scatter(x, y, c=kmeans.labels_) """


""" print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)
 """

#prediction, actual = test(1, x_test, x_train, y_train, y_test)
#print(confusion_matrix(prediction, actual, 10))

def sort_values(values, labels):
    #Sorts values by class [0-9]
    #print(values.shape)
    sorted_labels = np.argsort(labels)
    sorted_values = [values[i] for i in sorted_labels]
    sorted_values = np.array(chunks(sorted_values, 60000)).reshape(60000, 784)
    count = Counter(labels)
    #print(count.most_common(10))

    #for i in range(5):
    #    display_image(sorted_values[5922+i].reshape(28, 28))
    return sorted_values, count

#sorted_x_train = sort_values(x_train, y_train)


#prediction, actual = test(1, x_test, x_train, y_train, y_test)
#print(confusion_matrix(prediction, actual, 10))

def sort_values(values, labels):
    #Sorts values by class [0-9]
    #print(values.shape)
    sorted_labels = np.argsort(labels)
    sorted_values = [values[i] for i in sorted_labels]
    sorted_values = np.array(chunks(sorted_values, 60000)).reshape(60000, 784)
    count = Counter(labels)
    #print(count.most_common(10))

    #for i in range(5):
    #    display_image(sorted_values[5922+i].reshape(28, 28))
    return sorted_values, count

#sorted_x_train = sort_values(x_train, y_train)

# This might be vary wrong, a new try is in progress
def cluster(values, labels):
    sorted_values, count = sort_values(values, labels)

    # Calculating indices
    starts = 0
    ends = count[0]

    # 2D array for saving the centers
    center_classes = np.zeros((10,64))

    # Every class
    for i in range(10):
        # 1D array for the centers in each cluster
        center_class = np.zeros((64))
        values_one_class = sorted_values[starts:ends]
        starts = ends
        ends = starts + count[i]

        subarray_length = int(len(values_one_class) // 64)
        subarrays = np.array_split(values_one_class[0:subarray_length+1], 64)
        
        # Every cluster for one clas(cluster[k]-mean)s
        for k in range(64):
            center = np.mean(subarrays[k])
            center_class[k] = center
        center_classes[i] = center_class

    print(center_classes.shape)

    return center_classes

cluster(x_train, y_train)



# Part 2 - Clustering test


#for i in range(len(prediction)):
#    print(f'Predicted: {prediction[i]}, Actual: {actual[i]}\n')