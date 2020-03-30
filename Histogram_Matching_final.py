import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
def my_hist_match(image,desired_hist):

    rows, cols = image.shape
    print("Original Image Dimensions: ",rows,cols)

    #desired_hist = desired_hist * (rows*cols)

    cdf = np.zeros(256,np.uint32) #cumulative distribution function
    new_hist = np.zeros(256, np.uint8)
    new_image = np.zeros((rows,cols),np.uint8)

    for i in range(rows):
        for j in range(cols):
            cdf[image[i][j]] += 1 #calculate cdf
    #Calculate Histogram through Formula
    histogram = cdf / (rows * cols)
    histogram = histogram * 255
    for i in range(1,256):
        histogram[i] = histogram[i] + histogram[i-1]
    histogram = np.round(histogram)
    histogram = histogram.astype("uint8")
    plt.figure(1)
    plt.subplot(5, 1, 1)
    plt.title('Original Histogram')
    plt.plot(histogram)
    #print(histogram)
    #print(desired_hist)
    for j in range(256):

        nearest = desired_hist[min(range(len(desired_hist)), key=lambda i: abs(desired_hist[i] - histogram[j]))]
        find = np.where(desired_hist == nearest)
        find = np.asarray(find).flatten()
        index = min(find)
        new_hist[j] = index

    for i in range(256):
        a = np.where(image == i)
        a = np.asarray(a)
        for j in range(a.shape[1]):
            new_image[a[0][j]][a[1][j]] = new_hist[i]

    return(new_image,new_hist)

def desired_hist():

    x = np.arange(256).astype("float")# cumulative distribution function
    f = np.zeros(256,np.float)
    for i in range(256):
        f[i] = (x[i])
    #f = x / 256
    f = f / np.sum(f)
    print(f)
    # eğer bu değilse alt tarafı sil sadece 256x256 ile çarp
    plt.subplot(5, 1, 3)
    plt.title('Desired Histogram')
    plt.plot(f)

    f = f * 256
    for i in range(1, 256):
        f[i] = f[i] + f[i - 1]
    f = np.round(f)
    f = f.astype("uint8")

    #f = f * (256*256)
    return f

if __name__ == "__main__":

    image = cv2.imread("cameraman.tif", 0)
    desired_hist = desired_hist()

    matched_image,new_hist = my_hist_match(image,desired_hist)



    plt.subplot(5, 1, 5)
    plt.title('New Matched Histogram')
    plt.plot(new_hist)


    cv2.imwrite("pout_matched.png", matched_image)
    cv2.imshow("original",image)
    cv2.imshow("matched",matched_image)

    plt.show()
    cv2.waitKey()

