import numpy as np
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing import image
import matplotlib.pyplot as plt
from math import atan2, cos, sin, sqrt, pi,tan
import cv2 as cv
from tensorflow import keras



def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    ## [visualization]

    ### Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return -int(np.rad2deg(angle)) - 90
def check(img):
    if img is None:
        print("Error: File not found")
        exit(0)
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 3700 or 100000 < area:
            continue

        # Draw each contour only for visualisation purposes
        cv.drawContours(img, contours, i, (0, 0, 255), 2)

        # Find the orientation of each shape
        h=getOrientation(c, img)
        return h
model = keras.models.load_model(r'C:\Users\murar\Downloads\model.h5')
img1 = cv.imread(r"C:\Users\murar\Downloads\IMG_20230430_210438.jpg")
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
plt.imshow(img1)
plt.show()
m,n,v=img1.shape

h=0.0
non=0
c=0

ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
for q in range(0,m):
    for e in range(0,n):
        if (thresh1[q][e] > 100):
            non = non + 1
        else:
            c = c + 1
for i in range(0,m,120):
    for j in range(0,n,120):
        a = img1[i:i + 120, j:j + 120]
        b = np.expand_dims(a, axis=0)
        if(model.predict(b)>=0.5):
            k=check(img1[i:i+120,j:j+120])
            if k!=None and abs(sin(k))> 0.5:
                h+=120/abs(sin(k))
            else:
                h+=120
print("The resolution of the image is: ")
print(m,n,v)
print("The number of pixels in non cracked and cracked portion are: ")
print(non,c)
print("The length of the crack is: ")
print(h)
print("percentage of crack area is ")
print(c*100/(non+c))
print("The width of the crack is:")
print(c/h)





