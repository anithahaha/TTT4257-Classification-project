# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Constants
imagefile = 'train_images.bin'
labelfile = 'train_labels.bin'
txtfile = 'data_train.txt'

# Open binary file and and add data to txt
def convert(img_file, label_file, txt_file, n_images):
    lbl_f = open(label_file, "rb")   # labels (digits)
    img_f = open(img_file, "rb")     # pixel values
    txt_f = open(txt_file, "w")      # output file

    img_f.read(16)   # discard header
    lbl_f.read(8)    # discard header

    for i in range(n_images):   # number images 
        lbl = ord(lbl_f.read(1))  # get label
        for j in range(784):  # get 784 pixel values
            val = ord(img_f.read(1))    #ord gets unicode value
            txt_f.write(str(val) + "\t") 
    txt_f.write(str(lbl) + "\n")
    img_f.close(); txt_f.close(); lbl_f.close()

# Displays one image from txt file with index idx
def display_from_file(txtfile, idx):
    # Find index for loading data.
    idx_l = 784*idx
    idx_h = 784*(idx+1)

    # Load data from txt
    all_data = np.loadtxt(txtfile, delimiter="\t", usecols=range(idx_l, idx_h), dtype=np.int64)

    # Create image array with shape (28,28)
    image = all_data.reshape(28,28)

    # Display image
    plt.imshow(image, cmap='gray')
    plt.show()
    #plt.imsave('figures/testbilde.png', image, cmap="gray")

#convert(imagefile, labelfile, txtfile, 60000)
display_from_file(txtfile, idx=55)
