import numpy as np
# from scipy.misc import imread
import matplotlib.pyplot as plt
import cv2

from srm_funs import *


def srm_img_grad(I):
    """
    srm_img_grad: outputs the x-derivative  and y-derivative of the input I. If 
    I is 3D, then derivatives of each channel are available in xd and yd.
    """

    Ix = np.zeros(np.shape(I))
    Iy = np.zeros(np.shape(I))

    D = np.shape(I)[2]

    sob = np.array([[-1, 9, -45, 0, 45, -9, 1]])/60
    # print(Ix.shape)
    # print(sob)
    # print(np.transpose(sob))
    # print(D)

    for i in range(0, D):
        Ix[:, :, i] = cv2.filter2D(src = I[:, :, i], ddepth = -1, kernel = sob, borderType = cv2.BORDER_REPLICATE)
        Iy[:, :, i] = cv2.filter2D(src = I[:, :, i], ddepth = -1, kernel = np.transpose(sob), borderType = cv2.BORDER_REPLICATE)
        
    return Ix, Iy


class SRM:
    def __init__(self, image, Q = 8):
        self._height = image.shape[0]
        self._width = image.shape[1]
        if image.ndim == 3:
            self._depth = image.shape[2]
        else:
            self._depth = 1

        self._n = self._width * self._height
        self._image = image
        
        self._logdeta = 2.0 * np.log(6.0 * n)
        self._q = Q


if __name__ == "__main__":
    print("SRM begins!")

    #  test special

    # import cv2
    # import argparse
    # # create the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument('-i', '--image', required = True, help = 'Path to the input image')
    # args = vars(ap.parse_args())
 
    # # read the image
    image = cv2.imread("lena.png", cv2.IMREAD_COLOR) # b, g, r

    image = image.astype(np.float)
    print(image.dtype)
    
    # Smoothing the image, comment this line if you work on clean or synthetic image.
    h = fspecial("gaussian", (3, 3), 1)
    image_fil = cv2.filter2D(image, ddepth = -1, kernel = h, borderType = cv2.BORDER_REFLECT)
    

    b, g, r = cv2.split(image)
    image_plt = cv2.merge([r, g, b])
    
    b, g, r = cv2.split(image_fil)
    image_fil_plt = cv2.merge([r, g, b])


    plt.subplot(1,2,1), plt.imshow(image_plt)
    plt.subplot(1,2,2), plt.imshow(image_fil_plt)

    plt.savefig("filtered_image.jpg")

    # compute image gradient
    print(image_fil.dtype)
    Ix, Iy = srm_img_grad(image_fil)

    print(Iy[0 : 7, 0 : 7, 0])
    print(Iy[0 : 7, 0 : 7, 1])
    print(Iy[0 : 7, 0 : 7, 2])


    plt.subplot(1,2,1), plt.imshow(Ix)
    plt.subplot(1,2,2), plt.imshow(Iy)
    plt.savefig("image_gradients.jpg")
    
    Ix = np.max(np.abs(Ix), axis = 2)
    Iy = np.max(np.abs(Iy), axis = 2)
    plt.subplot(1,2,1), plt.imshow(Ix)
    plt.subplot(1,2,2), plt.imshow(Iy)
    plt.savefig("image_max_gradients.jpg")



    
    

    # # apply the 3x3 median filter on the image
    # processed_image = cv2.medianBlur(image, 3)
    # # display image
    # cv2.imshow('Median Filter Processing', processed_image)
    # # save image to disk
    # cv2.imwrite('processed_image.png', processed_image)
    # # pause the execution of the script until a key on the keyboard is pressed
    # cv2.waitKey(0)

