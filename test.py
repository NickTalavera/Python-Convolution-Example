import numpy as np
# import cv2

# img_file = cv2.imread('F:\CVIP Projects\Project1/lena_gray.jpg', 0)

# pad_img = cv2.copyMakeBorder(img_file, 1,1,1,1, cv2.BORDER_CONSTANT, value = 0)
height, width = pad_img.shape

def twoD_convolution():
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    convolution_x = np.zeros((height, width))
    convolution_y = np.zeros((height, width))
    convolution_mag = np.zeros((height, width))

    for i in range(1, height-1):
       for j in range(1, width-1):
           convolution_x[i][j] = np.sum(pad_img[i-1:i+2,j-1:j+2]*Gx)
           convolution_y[i][j] = np.sum(pad_img[i-1:i+2,j-1:j+2]*Gy)

    convolution_x = convolution_x/convolution_x.max()
    convolution_y = convolution_y/convolution_y.max()
    convolution_mag = np.sqrt((convolution_x*convolution_x) + (convolution_y*convolution_y))
    # cv2.imshow('2D_x', convolution_x)
    # cv2.imshow('2D_y', convolution_y)
    # cv2.imshow('2D_mag', convolution_mag)

def oneD_convolution():
    Gx_v = np.array([[1],[2],[1]])
    Gx_h = np.array([-1,0,1])
    Gy_v = np.array([[-1],[0],[1]])
    Gy_h = np.array([1,2,1])

    convolution_x = np.zeros((height, width))
    convolution_y = np.zeros((height, width))
    convolution_mag = np.zeros((height, width))
    median_x = np.zeros((height, width))
    median_y = np.zeros((height, width))

    for i in range(1, height-1):
        for j in range(1, width-1):
            median_x[i][j] = np.sum(pad_img[i, j-1:j+2]*Gx_h)
            median_y[i][j] = np.sum(pad_img[i, j-1:j+2]*Gy_h)

    for i in range(1, height-1):
        for j in range(1, width-1):
            x1 = median_x[i-1][j]
            x2 = median_x[i][j]
            x3 = median_x[i+1][j]
            array_x = np.array([[x1],[x2],[x3]])
            convolution_x[i][j] = np.sum(array_x*Gx_v)
            y1 = median_y[i-1][j]
            y2 = median_y[i][j]
            y3 = median_y[i+1][j]
            array_y = np.array([[y1],[y2],[y3]])
            convolution_y[i][j] = np.sum(array_y*Gy_v)

    convolution_x = convolution_x/convolution_x.max()
    convolution_y = convolution_y/convolution_y.max()
    convolution_mag = np.sqrt((convolution_x*convolution_x) + (convolution_y*convolution_y))
    convolution_mag = convolution_mag/convolution_mag.max()
    # cv2.imshow('1D_x', convolution_x)
    # cv2.imshow('1D_y', convolution_y)
    # cv2.imshow('1D_mag', convolution_mag)


# twoD_convolution()
oneD_convolution()
