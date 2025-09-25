import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
Saves segmented images to array form to save time
"""

def imgsToGray(imgFolderPath):
    firstFileName = os.path.join(imgFolderPath, os.listdir(imgFolderPath)[0])
    imgArray = cv2.imread(firstFileName, cv2.IMREAD_GRAYSCALE) / 255.0
    imgArray = imgArray[..., np.newaxis]
    for file in os.listdir(imgFolderPath)[1:]:
        fileName = os.path.join(imgFolderPath,file)
        img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) / 255.0
        img = img[...,np.newaxis]
        imgArray = np.append(imgArray,img,axis=2)
    return np.transpose(imgArray, (2,0,1)) # Return shape (numImgsInFolder, 120, 120)

def imgsToRGBArray(imgFolderPath):
    firstFileName = os.path.join(imgFolderPath,os.listdir(imgFolderPath)[0])
    imgArray = cv2.imread(firstFileName) / 255.0
    imgArray = imgArray[..., np.newaxis]
    for file in os.listdir(imgFolderPath)[1:]:
        fileName = os.path.join(imgFolderPath,file)
        img = cv2.imread(fileName) / 255.0
        img = img[...,np.newaxis]
        imgArray = np.append(imgArray,img,axis=3)
    # Right now img array is in shape (120,120,3, numImgsInFolder)
    return np.transpose(imgArray, (3, 0, 1, 2)) # This changes it to (numImgsInFolder,120,120,3) which is what tensorflow expects

def noise(array):
    noise_factor = 0.5
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1, size=array.shape
    ) # loc is the center of the normal distrib, scale is the SD

    return np.clip(noisy_array, 0.0, 1.0)
    # Makes sure the highest value the noise can be is 1 and lowest is 0 so the entire image
    # is between 0 and 1 still

trainPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Original\Training"
testPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Original\Validation"
curatedTestPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Original\CuratedValidation"

trainArray = imgsToRGBArray(trainPath)
trainNoiseArray =noise(trainArray)
testArray = imgsToRGBArray(testPath)
curatedTestArray = imgsToRGBArray(curatedTestPath)
grayTrainArray = imgsToGray(trainPath)
grayTestArray = imgsToGray(testPath)
curGrayTestArray = imgsToGray(curatedTestPath)

# Just to see how noisy the images turn out
"""def display(array1, array2):
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(120, 120,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(120, 120,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

display(trainArray,trainNoiseArray)"""


np.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\trainArray.npy", trainArray)
np.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\testArray.npy", testArray)
np.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\trainNoiseArray.npy",trainNoiseArray)
np.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\curatedTestArray.npy",curatedTestArray)
np.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\grayTrainArray.npy", grayTrainArray)
np.save(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\grayTestArray.npy", grayTestArray)