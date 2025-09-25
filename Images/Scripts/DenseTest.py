import keras.models
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

"""
Testing it and visualizing results
"""
autoencoder = keras.models.load_model(r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\TrainedModels\denseAutoencoder.keras")
#testArrayPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\curatedTestArray.npy"
testArrayPath = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Img Arrays\grayTestArray.npy"
x_test = np.load(testArrayPath) # (numImgsInFolder, 120, 120) These are gray scale only


# We can test it by setting an error threshold. Only images with reconstruction
# errors greater than that threshold should be displayed in a window

# Ordering the images based on error from greatest to least
numImgs = np.shape(x_test)[0]
lossList = []
reconList = [] # list of reconstructions produced by the model

for i,x in enumerate(x_test):
    x = x[np.newaxis, ..., np.newaxis] # (feed it in with shape (1, 120, 120, 1)
    print(np.shape(x))
    print(f"Image number: {i+1}/{numImgs}")
    loss = 1000*autoencoder.evaluate(x,x) # multiply it by 1000 so it's not in the decimal range
    reconstruction = autoencoder.predict(x) # only need input for prediction, no label
    reconstruction = np.squeeze(reconstruction)
    reconList.append(reconstruction)
    lossList.append(loss)

# Sorting all lists in parallel by loss


lossArray = np.array(lossList)
reconArray = np.array(reconList)

idxSort = np.argsort(lossArray) # Sorts in ascending by default
xSorted = x_test[idxSort]
reconListSorted = reconArray[idxSort]
lossSorted = lossArray[idxSort]


# show the imgs that have the highest error from greatest to least
numToDisplay = 25
cols = 5
rows = int(np.ceil(numToDisplay / cols))


"""
Displaying the images that were reconstructed most accurately (should expect no malaria cells here)
"""
fig1, axs1 = plt.subplots(rows, cols)
fig1.suptitle(f"Top {numToDisplay} Lowest Error Reconstructions")
axs1 = axs1.flatten() # into an easily iterable col vec rather than an array form

for idx, (loss, recon) in enumerate(zip(lossSorted,reconListSorted)):
    if idx >= numToDisplay:
        break
    axs1[idx].set_title(f"Loss {loss:.2f}")
    axs1[idx].imshow(recon)

fig2, axs2 = plt.subplots(rows, cols)
fig2.suptitle(f"Top {numToDisplay} Lowest Error Original Images")
axs2 = axs2.flatten()

for idx, (loss, xIdx) in enumerate(zip(lossSorted, idxSort)):
    if idx >= numToDisplay:
        break
    axs2[idx].set_title(f"Loss {loss:.2f}")
    axs2[idx].imshow(x_test[xIdx])


"""
Displaying the images that were reconstructed most poorly (should see a lot of malaria cells here)
"""
fig3, axs3 = plt.subplots(rows, cols)
fig3.suptitle(f"Top {numToDisplay} Highest Error Reconstructions")
axs3 = axs3.flatten() # into an easily iterable col vec rather than an array form

for idx, (loss, recon) in enumerate(zip(lossSorted[::-1],reconListSorted[::-1])):
    if idx >= numToDisplay:
        break
    axs3[idx].set_title(f"Loss {loss:.2f}")
    axs3[idx].imshow(recon)

fig4, axs4 = plt.subplots(rows, cols)
fig4.suptitle(f"Top {numToDisplay} Highest Error Original Images")
axs4 = axs4.flatten()

for idx, (loss, xIdx) in enumerate(zip(lossSorted[::-1], idxSort[::-1])):
    if idx >= numToDisplay:
        break
    axs4[idx].set_title(f"Loss {loss:.2f}")
    axs4[idx].imshow(x_test[xIdx])

plt.show()
