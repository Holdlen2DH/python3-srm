import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

def fspecial(type_name, p2 = None, p3 = None):
    """
    create prdefined 2-D filter of matlab style

    Gaussin low pass filter.
    fspecial("gaussian", p2 = hsize, p3 = sigma)
    """

    if type_name is "gaussian":
        print("Gaussian lowpass filer.")
        hsize = p2
        sigma_val = p3
        print("hsize = [%d, %d], sigma = %.2f\n"%( hsize[0], hsize[1], sigma_val))
        
        siz = (hsize - 1.0)/2
        std = sigma_val
        x, y = np.meshgrid(np.linspace(siz))
    else:
        print("type name does not match one of these strings \" gaussian\"")

    return
def fspecial_test():
    """
    test functionality of fspecial.
    """
    fspecial("gaussian", p2 = [3, 3], p3 = 0.5)

    return

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

    def img_grad(self):
        a = 1



if __name__ == "__main__":
    print("SRM begins!")

    #  test special
    fspecial_test()

    # import cv2
    # import argparse
    # # create the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument('-i', '--image', required = True, help = 'Path to the input image')
    # args = vars(ap.parse_args())
 
    # # read the image
    # image = cv2.imread("lena.png")
    # # apply the 3x3 median filter on the image
    # processed_image = cv2.medianBlur(image, 3)
    # # display image
    # cv2.imshow('Median Filter Processing', processed_image)
    # # save image to disk
    # cv2.imwrite('processed_image.png', processed_image)
    # # pause the execution of the script until a key on the keyboard is pressed
    # cv2.waitKey(0)

