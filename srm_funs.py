import numpy as np

def fspecial(type_name, p2 = None, p3 = None):
    """
    create prdefined 2-D filter of matlab style

    Gaussin low pass filter.
    fspecial("gaussian", p2 = hsize, p3 = sigma)

    Have known how to filter an image, stop writing this function.
    """

    if type_name is "gaussian":
        hsize = np.array(p2)
        sigma_val = p3
        
        siz = (hsize - 1)/2
        std = sigma_val
        x, y = np.meshgrid(np.arange(-siz[1], siz[1] + 1, step = 1), np.arange(-siz[0], siz[0] + 1, step = 1))
        

        arg = -(x * x + y * y)/(2 * std * std)
        h = np.exp(arg)

        sumh = np.sum(h)
        if sumh is not 0:
            h = h/sumh

    else:
        print("type name does not match one of these strings \" gaussian\"")
        h = None

    return h
def fspecial_test():
    """
    test functionality of fspecial.
    """
    h = fspecial("gaussian", p2 = [3, 3], p3 = 1)
    print(h)
    return

if __name__ == "__main__":
    print("SRM begins!")

    #  test special
    fspecial_test()