import cv2
import numpy as np
img = cv2.imread("1.tif")

img = cv2.medianBlur(img,5)
_, threshold = cv2.threshold(img,194, 255, cv2.THRESH_BINARY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
 
cv2.imwrite("4x2.tif", threshold)
cv2.imwrite("4x.tif", mean_c)
#cv2.imwrite("4x3.tif", gaus)
 
img3 = cv2.imread('4x2.tif', cv2.IMREAD_GRAYSCALE)
n_white_pix = np.sum(img3 == 255)
print('Number of white pixels:', n_white_pix)

black_px = np.sum(img3 == 0)
print('Number of black pixels:', black_px)

print('Confluency is : ' ,black_px*3 *100/ (black_px+n_white_pix)) 
