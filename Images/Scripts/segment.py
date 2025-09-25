"""
Quantifying Pathologist Tunnel Vision Using Machine Learning
A Proof of Concept Using Intracellular Parasites
PIs: Dr. Behling
This code functions to segment a given whole field image of a blood smear (ideally thin) under 100X magnification
It automatically segments the given image with decent accuracy to produced cropped 120x120 pixel images of each cell
The segmentation procedure relies on the following pipeline:
1. Blur image to ignore dust
2. Threshold
3. Distance Transform and Peak selection
4. Watershed method
5. Cropping based on computed centroids

Areas of improvement: there is variability in the peak selection process, but, for the
sake of time, this was left for future work

Coded by: Adam Dalkilic
9-2-24
"""

import cv2
import numpy as np
import os

# Helper functions
def extractImages(path):
    # Input is a folder of pngs or jpgs
    # output is a list of dictionaries of the images: imageListDict
    # The attributes of each component of imageListDict are as follows:
    # "Name":filename, the name of the file the image was read from
    # "Image":image, the image as an opencv matrix

    imageListDict= []
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path,filename))
        imageDict = {"Name":filename,"Image":image}
        imageListDict.append(imageDict)
    return imageListDict

def segmentImages(imageListDict, savePath, cropSize, show=True):
    # Input: list of whole field images dictionaries (the output of the extractImages function)
    # Output: list of segment image dictionaries. Each component of this dictionary has the following
    # attributes:
    # "SlideName": str, the name of the whole field image file this image was cropped from
    # "CellNumber": int, the label of the cell
    # "Centroid": (centerx,centery), the two coords representing the center of the cropped image
    # The cropped images will be 120 x 120 for example if cropSize = 120

    for i, imageDict in enumerate(imageListDict):
        imageName = imageDict['Name']
        image = imageDict['Image']
        H = np.shape(image)[0]
        W = np.shape(image)[1]
        imgArea = H*W

        # Convert to gray scale
        imageGray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        """display = cv2.resize(imageGray, (W // 2, H // 2))
        cv2.imshow("Gray",display)"""

        # Blur the image slightly to blend out dust
        blur = cv2.GaussianBlur(imageGray ,(21,21),0)
        """display = cv2.resize(blur, (W // 2, H // 2))
        cv2.imshow("Blur",display)"""

        # Threshold the image via Otsu's method to obtain foreground and background
        # Otsu's method estimates a threshold and returns that estimated value as threshVal
        # thresh is the newly obtained binary image. We pass 0 for thresh because it calculates it itself
        threshVal, threshInv = cv2.threshold(blur, 0, 255,cv2.THRESH_OTSU)
        thresh = 255 - threshInv
        """display = cv2.resize(thresh, (W // 2, H // 2))
        cv2.imshow("Thresh", display)"""

        # Now we have to fill in the holes created by Otsu's algorithm
        wavefront = np.zeros(np.shape(threshInv)).astype(np.uint8)
        wavefront = np.pad(wavefront,1,mode='constant',constant_values=255)
        threshP = np.pad(thresh,1)
        threshP = 255-threshP  # After this, the bg is white (255), the fg is black (0)
        notDone = True
        while notDone:
            kernel = np.ones((3,3))
            prevWF = np.copy(wavefront).astype(np.uint8)
            wavefront = cv2.dilate(wavefront,kernel)
            wavefront= cv2.bitwise_and(wavefront,threshP)
            #cv2.imshow("wave front",wavefront)
            #cv2.imshow("original",threshP)
            if (prevWF == wavefront).all():
                # Remove the pad
                holesFilled = wavefront[1:-1,1:-1]
                holesFilled = 255 - holesFilled
                notDone = False
        """display = cv2.resize(holesFilled, (W // 2, H // 2))
        cv2.imshow("Holes Filled", display)"""


        # Now we can perform distance transform on the image without holes
        dist = cv2.distanceTransform(holesFilled, cv2.DIST_L2, 5)
        """dist_Norm = cv2.normalize(dist,None,0,255,cv2.NORM_MINMAX)# only for display
        dist_Norm = dist_Norm.astype(np.uint8)
        display = cv2.resize(dist_Norm, (W // 2, H // 2))

        cv2.imshow("Dist",display)"""
        # Now we threshold the distance transform to only take the brightest pixels
        # These pixels are our seeds for the watershed algorithm
        _, seeds = cv2.threshold(dist, 0.33 * dist.max(), 255, 0)  # Taking the brightest pixels as
        seeds = np.uint8(seeds)

        """display = cv2.resize(seeds, (W // 2, H // 2))
        cv2.imshow("Seeds",display)
        cv2.waitKey(0)"""
        unknown = cv2.subtract(holesFilled,seeds)

        # Now we can use the connected components built in to extract the number of cells (a.k.a num of connected components)
        # and the markers, which are the centroids of these cells
        numComponents, pixelSeeds = cv2.connectedComponents(seeds)
        pixelSeeds = pixelSeeds + 1 # this makes the sure background 1 instead of 0, so it's treated as a basin
        pixelSeeds[unknown==255] = 0
        # That way the watershed algorithm won't keep expanding if there are few objects


        # We feed in our original colored image with the pixelSeeds
        markers = cv2.watershed(image, pixelSeeds)

        # We get an output that is an array of the same size as the original whole field image
        # Each pixel is labeled -1 if it is a border, or it is labeled with an integer representing
        # which object it is in. For example, if I'm looking for the pixels in region 20, then all
        # the pixels in the array that are of this segment will have a value of 20.
        # The total number of segments is numComponents + 1 (Region 0 is also considered a segment)

        # Computing centroids of each segment, area, and filtering out garbage segments
        metadata = []
        for label in range(1,numComponents):
             # a list of dictionaries meant to store metadata for each cell

            coordinateList = np.argwhere(markers == label) # Returns the coordinates of the pixels (row,col) that correspond to the label
            yL, xL = zip(*coordinateList)
            minX = min(xL)
            maxX = max(xL)
            minY = min(yL)
            maxY = max(yL)
            centroid = ((minX + maxX)//2,(minY + maxY)//2)
            area = len(coordinateList)

            if area > imgArea//300:  # threshold for removing over-sized segments ~3.33% of original image size
                for x, y in zip(xL, yL):
                    markers[y][x] = 0 # 0is our garbage bin, a new label we create
                continue

            if area < imgArea//1000:  # threshold for removing small things like platelets and dust ~0.1% of original image size
                for x, y in zip(xL, yL):
                    markers[y][x] = 0  # 0is our garbage bin, a new label we create
                continue

            else:
                cx = centroid[0]
                cy = centroid[1]

                if (cx - cropSize//2 < 0) or (cy - cropSize//2 < 0) or (cx + cropSize//2 > W) or (cy + cropSize//2 > H):
                    print(f"\nCell {label} was too close to the border. Skipping . . .")
                    continue

                else:
                    # Time to crop the object
                    crop = image[cy - cropSize // 2:cy + cropSize // 2,
                           cx - cropSize // 2:cx + cropSize // 2]  # I can't use minX and maxX etc because they
                    # might've divide unevenly. It's better to just add 59 to each side of the centroid

                    # Appending metadata
                    metadata.append({"Cell Number": label,"Slide Number": imageName,"Centroid": centroid,"Area": area})
                    fName = os.path.join(savePath,f"Slide-{imageName[:-4]}_Cell_{label}.jpg")
                    cv2.imwrite(fName,crop)

        # Just for visualization
        if show == True:
            # show the watershed boundaries
            # Add text at the centroid of each label

            for dict in metadata:

                centroid = dict["Centroid"]
                label = dict["Cell Number"]
                area = dict["Area"]
                cv2.putText(image, (str(label)), centroid, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255,0,0),thickness = 1)
            image[markers == -1] = (255,0,0)# Make the watershed boundaries visible

            display = cv2.resize(image, (W//2,H//2))
            cv2.imshow("Final image", display)
            cv2.waitKey(0)



trainingWholeFieldImgs = extractImages(
    r"/Proof Of Concept-Intracellular Parasite Detection/Whole Field Images/Training")
validationWholeFieldImgs = extractImages(
    r"/Proof Of Concept-Intracellular Parasite Detection/Whole Field Images/Validation")
savePathTrain = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Training"
savePathValidation = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\Segmented Images\Validation"
savePathValidation = r"C:\Users\dalki\PycharmProjects\Pathology\Proof Of Concept-Intracellular Parasite Detection\GarbageBin"
#segmentImages(trainingWholeFieldImgs,savePathTrain,cropSize=120,show=True)
segmentImages(validationWholeFieldImgs,savePathValidation,cropSize=120,show=True)