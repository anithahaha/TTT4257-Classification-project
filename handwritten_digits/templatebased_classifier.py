# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

"""
x_train = 
y_train =

x_test =
y_test = 
hi
"""

def chunks(array, L):
    array = np.array_split(array, L)
    return array

def euclidean_distance(qi, pi):
    #function calculating euclidean distance
    return np.sqrt(np.sum((qi-pi)**2))

def kNN(k, x, x_train, y_train):
    distance = []
    #calculate distance between each training value
    for ref in x_train:
        distance.append(euclidean_distance(x, ref))
    #calculate index value of k NN
    k_index = np.argsort(distance)[:k]
    k_label = [y_train[i] for i in k_index]
    #perform majority vote
    count = Counter(k_label)
    pred = count.most_common(1)
    return pred

def test(k, x_test, x_train, y_train, y_test):
    prediction = []
    actual = y_train
    for i in range(len(x_test)):
        pred_value = kNN(k, x_test[i], x_train, y_train)
        prediction.append(pred_value)
    return prediction, actual

def error_rate():
    """Calculates error rate of classifier"""
    return;

def confusion_matrix():
    """Calculates and displays confusion matrix"""
    return;

def display_image():
    """Displays image (both classified and misclassified)"""
    return;

def clustering():
    return;
