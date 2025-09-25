import cv2
import os
import numpy as np
from keras import layers
from keras.models import Model
import tensorflow as tf


curatedPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\CuratedValidation"
trainingPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Training"
validationPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Validation"


def ssim_loss(y_true, y_pred):
    ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
    # As loss we want to minimize (1 - SSIM)
    return 1.0 - tf.reduce_mean(ssim_val)


kernelNum = 32 # number of kernels that the model is learning. Each kernel highlights a different set of
# features of the image
kernelSize = (4,4) # the pixel size of the kernel being passed we keep it divisible by stride to avoid checkerboard effect
input = layers.Input(shape = (120,120,3))

# Encoder
x = layers.Conv2D(kernelNum ,kernelSize, activation = "relu", padding="same")(input) # Creating 32 feature maps: (120,120,32)
# Padding prevents loss of edge pixels, which normally occurs; for example prevents 120,120 --> 118,118
x = layers.MaxPooling2D((4,4), padding="same")(x)  # Dividing height and width by 2, 2 --> (60,60,32)
x = layers.Conv2D(kernelNum ,kernelSize,activation = "relu", padding="same")(x) # (60,60,32)
x = layers.MaxPooling2D((2,2), padding="same")(x) # (30,30,32)

# Decoder
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(kernelNum,kernelSize,padding="same",activation="relu")(x)
x = layers.UpSampling2D((4,4))(x)
x = layers.Conv2D(kernelNum,kernelSize,padding="same",activation="relu")(x)
x = layers.Conv2D(3, kernelSize, activation="sigmoid",padding="same")(x)

autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

trainArrayPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\trainArray.npy"
trainNoisePath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\trainNoiseArray.npy"
testArrayPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\testArray.npy"

trainImgs = np.load(trainArrayPath)
trainNoiseImgs = np.load(trainNoisePath)
testImgs = np.load(testArrayPath)

# x is the input, y is the label but for autoencoders x=y
autoencoder.fit(x = trainNoiseImgs, y = trainImgs, epochs = 50, batch_size = 128, shuffle = True, validation_data= (testImgs, testImgs))
autoencoder.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\TrainedModels\Autoencoder.keras")