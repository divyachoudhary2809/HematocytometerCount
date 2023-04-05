import skimage
from PIL import Image 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from skimage import morphology 
 


def get_img(threshold_number,mask_number):
            img3 = np.where(imgs[threshold_number]==mask_number, th[threshold_number],7)    
            return img3,Image.fromarray(img3), Image.fromarray(img3).show()

    
    
img_ini = cv2.imread('hctyes2.tif')   
th =[[]] * 4
imgs = [[]] * 4    
reta,th[0] = cv2.threshold(img_ini,75,255,cv2.THRESH_BINARY)#160#187
ret,th[1] = cv2.threshold(img_ini,75,255,cv2.THRESH_BINARY)#180#190
reta,th[2] = cv2.threshold(img_ini,75,255,cv2.THRESH_BINARY)#200
reta,th[3] = cv2.threshold(img_ini,75,255,cv2.THRESH_BINARY)#220

for i in range(len(th)):
            img = skimage.morphology.label(th[i], neighbors=None, background=None, return_num=False, connectivity=None)
            imgs[i] =img


x= get_img(1,65) 
#y =x[0]
#
import scipy.misc
scipy.misc.imsave('outfile.jpg', x[0])



import cv2
import numpy as np
img = cv2.imread("outfile.jpg")

img = cv2.medianBlur(img,5)
_, threshold = cv2.threshold(img,75, 255, cv2.THRESH_BINARY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
 
cv2.imwrite("4x2.tif", threshold)
 
 
img3 = cv2.imread('4x2.tif', cv2.IMREAD_GRAYSCALE)
n_white_pix = np.sum(img3 == 255)
n_pix=n_white_pix
print('Number of white pixels:', n_white_pix)

black_px = np.sum(img3 == 0)
print('Number of black pixels:', black_px)

print('cell area is : ' ,n_white_pix) 



import cv2
import numpy as np
img = cv2.imread("hctyes2.tif")

img = cv2.medianBlur(img,5)
_, threshold = cv2.threshold(img,75, 255, cv2.THRESH_BINARY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
 
cv2.imwrite("4x2.tif", threshold)
 
 
img3 = cv2.imread('4x2.tif', cv2.IMREAD_GRAYSCALE)
n_white_pix = np.sum(img3 == 255)
print('Number of white pixels:', n_white_pix)

black_px = np.sum(img3 == 0)
print('Number of black pixels:', black_px)

print('number of cells is : ' ,n_white_pix/n_pix) 

