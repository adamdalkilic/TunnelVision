import cv2
import os
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt


def imgsToArray(imgFolderPath):
    firstFileName = os.path.join(imgFolderPath,os.listdir(imgFolderPath)[0])
    imgArray = cv2.imread(firstFileName,cv2.IMREAD_GRAYSCALE) / 255.0
    imgArray = imgArray[...,np.newaxis]
    for file in os.listdir(imgFolderPath)[1:]:
        fileName = os.path.join(imgFolderPath,file)
        img = cv2.imread(fileName,cv2.IMREAD_GRAYSCALE) / 255.0
        img = img[...,np.newaxis]
        imgArray = np.append(imgArray,img,axis=2)
    return np.swapaxes(imgArray,0,2)

def createModel():
    model = tf.keras.Sequential([
        keras.layers.Flatten(input_shape=(120,120,1)),
        keras.layers.Dense(120 ** 2, activation='relu'),
        keras.layers.Dense(1000, activation = 'relu'),
        keras.layers.Dense(100, activation = 'relu'), # Number of latent features
        keras.layers.Dense(1000, activation = 'relu'),
        keras.layers.Dense(120 ** 2, activation='sigmoid'),
        keras.layers.Reshape((120,120,1)) # Reshaping back to original image
    ])

    model.compile(optimizer='adam',
                  loss='mse') # mean squared error

    return model

# Packaging the training and validation data
trainingPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\trainArray.npy"
validationPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\testArray.npy"

x_train = np.load(trainingPath) # Original imports are 3 channel
x_test = np.load(validationPath)

print(np.shape(x_train))
print(np.shape(x_test))

x_train = np.average(x_train,axis=3) # Compress down to gray scale to reduce computation time
x_test = np.average(x_test,axis=3)

plt.imshow(x_test[0][:])
plt.show()
print(np.shape(x_train))
print(np.shape(x_test))

autoencoder = createModel()
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size = 50,
                shuffle=True,
                validation_data=(x_test, x_test))

"""
Saving the model
"""

autoencoder.save('denseFinal.keras')

