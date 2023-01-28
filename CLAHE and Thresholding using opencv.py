# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:01:34 2022

@author: Victor-PhD
"""

''' 
Why do we need Histogram Equalization?

Ans: Stretch the Histogram to span the entire range and make it more clearer
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Read the image as a gray level image. "0" stands for graylevel & "1" for color imgage
img = cv2.imread("3_in1.png", 0)

#For Histogram Equalization
eq_img = cv2.equalizeHist(img)

#Plot / visualize the image b/4 equalization
plt.hist(eq_img.flat, bins=10, range=(0, 255))
plt.xlim([0,256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
plt.savefig("3_in1HE.png")

'''
Observation: My Hstogram os "Skewed" towards the right, let's hope 
Histogram Equalization fixed this

'''
#Plot / visualize the image after Histogram equalization
plt.hist(eq_img.flat, bins=10, range=(0, 255))
#Yes, the image is now Equalized and stretches between 0 and 255

'''
Observe that the Histogram Equalized image appears darker  and has lots of
added noise. The size (in MB) is also larger as it considers the "Global Contrast"
and not just the local contrast of the image. Thus, we use Adaptive Histogram Eqaulaization
aka CLAHE. 

Contrast Limited AHE (CLAHE) is a variant of adaptive histogram equalization in which the 
contrast amplification is limited, so as to reduce this problem of noise amplification. 
In CLAHE, the contrast amplification in the vicinity of a given pixel value is given by the slope of the 
transformation function.

'''
#1. Read the image again
img = cv2.imread("3_in1.png", 0)

#2. Create a CLAHE Object(Arguments are Optional)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))# Read up the official documentation
cl_img = clahe.apply(img) # Apply CLAHE on the original image

#3. Display the images
#cv2.imshow("Equalized Image", eq_img)
cv2.imshow("CLAHE Image", cl_img)

#4. Show the CLAHE Histogram with bins = 100
#Plot / visualize the image after Histogram equalization
plt.hist(eq_img.flat, bins=10, range=(0, 255))
#Yes, the image is now Equalized and stretches between 0 and 255

#5. Save the Image & Destroy all windows
cv2.imwrite("CLAHEimage3_in1.png", cl_img)#Saves the image..Added by me
print("CLAHE Image written to file-system")
cv2.waitKey(0) #Waits forever until we close the window
cv2.destroyAllWindows() #Closes and exit

#-----------------------------------------------------------------------------------
# Even after CLAHE, the output image is still noisy. Let's go and clan them up

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("3_in1.png", 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img) 

#plt.hist(cl_img.flat, bins=100, range=(0, 255))

plt.hist(cl_img.flat, bins=100, range=(120, 255))

#Now apply BINARY thresholding after the 120 - 255 range
# Please refer the official documenation
ret, thresh1 = cv2.threshold(cl_img, 190, 150, cv2.THRESH_BINARY) #Unpack the two, just ignore the first argument (ret), otherwise it will throw an error
#ret, thresh2 = cv2.threshold(cl_img, 190, 255, cv2.THRESH_BINARY_INV) #Perform inverting operation

#cv2.imshow("Original Image", img)
#cv2.imshow("Binary Threshold 1", thresh1)
#cv2.imshow("Binary Threshold 2", thresh2)
#cv2.imwrite("BinaryThresh1.png", thresh1)#Saves the image..Added by me
#cv2.imwrite("BinaryThresh2.png", thresh2)
#print("CLAHE Image written to file-system")

#Now apply OTSU based thresholding after the 120 - 255 range
ret2, thresh3 = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Original Image 3-in1", img)
cv2.imshow("OTSU Threshold", thresh3)
# N/B: Unlike the hard coded value of 190 for binary thresholding, 
# OTSU found that the best value to separate them is at 132.0 
cv2.imwrite("OTSUThreshold.png", thresh3)
cv2.waitKey(0)
cv2.destroyAllWindows()






