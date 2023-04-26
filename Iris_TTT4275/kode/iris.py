# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

# Import data
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 

# Load the data
df = pd.read_csv('Iris_TTT4275/iris.data', names=columns)
df.head()
data = df.values

# Converting to numpy array
data = np.array(data)
#print(data.shape)
data_sorted = np.array([data[0:50], data[50:100], data[100:150]])

# Task 1
# a) Choose first 30 samples for training and last 20 for testing.
data_train = data_sorted[:, 0:30]
Y_train = data_train[-1]
X_train = data_train[0:-2]

print(Y_train.shape)
#testinghehe

data_test = data_sorted[:, 30:50]


#print(data_test.shape)

