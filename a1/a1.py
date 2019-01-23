import numpy as np
import cv2 as cv
import scipy.stats as st
from scipy import signal
from scipy.ndimage.filters import laplace
from scipy.ndimage.filters import gaussian_filter

def correlation(image, kernel, mode):
    kernel_x,kernel_y = kernel.shape
    if kernel_x % 2 ==0 or kernel_y % 2 == 0 or kernel_x != kernel_y:
        return "invalid kernel dimension"

    num_rows = image.shape[0]
    num_cols = image.shape[1]
    
    if len(image.shape) == 2:

        k = kernel.shape[0] / 2

        if mode == "valid":
            output = np.zeros((num_rows-2*k, num_cols-2*k))
        elif mode == "same":
            output = np.zeros((num_rows,num_cols))
            image = cv.copyMakeBorder(image,k,k,k,k, cv.BORDER_CONSTANT, value=0)
        elif mode == "full":
            offset = kernel.shape[0] - 1
            output = np.zeros((num_rows+offset, num_cols+offset))
            image = cv.copyMakeBorder(image,offset,offset,offset,offset,cv.BORDER_CONSTANT, value=0)

        new_num_rows = image.shape[0]
        new_num_cols = image.shape[1]

        for r in range(k, new_num_rows-k):
            for c in range(k, new_num_cols-k):
                patch = image[r - k:r + k+1, c - k:c + k+1]
                output[r-k,c-k] = np.sum(np.multiply(patch.flatten(), kernel.flatten()))

    else:
        k = kernel.shape[0] / 2

        if mode == "valid":
            output = np.zeros((num_rows-2*k, num_cols-2*k,3))
        elif mode == "same":
            output = np.zeros((num_rows,num_cols,3))
            image = cv.copyMakeBorder(image,k,k,k,k, cv.BORDER_CONSTANT, value=0)
        elif mode == "full":
            offset = kernel.shape[0] - 1
            output = np.zeros((num_rows+offset, num_cols+offset,3))
            image = cv.copyMakeBorder(image,offset,offset,offset,offset,cv.BORDER_CONSTANT, value=0)

        new_num_rows = image.shape[0]
        new_num_cols = image.shape[1]

        for r in range(k, new_num_rows-k):
            for c in range(k, new_num_cols-k):
                for channel in range(0, 3):
                    patch = image[r - k:r + k+1, c - k:c + k+1,channel]

                    output[r-k,c-k,channel] = np.sum(np.multiply(patch.flatten(), kernel.flatten()))

    return output




def question1():
    kernel = np.array([[-0.25,0,0.25],[1,0,1],[0.25,0,-0.25]],dtype='f')
    im = cv.imread('iris.jpg')

    resultA = correlation(im, kernel, "same")
    cv.imwrite('output_image_same.jpg', resultA)
    resultB = correlation(im, kernel, "full")
    cv.imwrite('output_image_full.jpg', resultB)
    resultC = correlation(im, kernel, "valid")
    cv.imwrite('output_image_valid.jpg', resultC)


def question2():
    x = cv.getGaussianKernel(3,3,ktype=cv.CV_64F)
    y = cv.getGaussianKernel(3,5,ktype=cv.CV_64F)
    gkernel = np.outer(x,y)
    print gkernel
    gkernel = np.flip(gkernel,0)
    gkernel = np.flip(gkernel,1)
    im = cv.imread('iris.jpg')
    result = correlation(im, gkernel, "same")
    cv.imwrite('output_2.jpg', result)


def question8():
    img = cv.imread('whereswaldo.jpg') 

    template = cv.imread('waldo.jpg') 
    w = template.shape[0]
    h=template.shape[1]

    res = cv.matchTemplate(img,template,cv.TM_CCORR_NORMED) 
    
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    cv.imwrite('Detected.jpg',img) 



def question7():
    img = cv.imread("portrait.jpg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    resultx = gaussian_filter(img_gray,3,order=[0,1])
    cv.imwrite("resultx.jpg",resultx)
    resulty = gaussian_filter(img_gray,3,order=[1,0])
    cv.imwrite("resulty.jpg",resulty)
    resultxy = gaussian_filter(img_gray,3,1)
    cv.imwrite("resultxy.jpg",resultxy)
    lap = cv.Laplacian(img_gray,cv.CV_64F,ksize=3)
    cv.imwrite("result_laplace.jpg",lap)

def question11():
    img = cv.imread("portrait.jpg",0)
    edges = cv.Canny(img,400,200)
    cv.imwrite("Canny_output.jpg",edges)



if __name__ == "__main__":
    # question1()
    # question7()
    question2()
    # question8()
    # question11()