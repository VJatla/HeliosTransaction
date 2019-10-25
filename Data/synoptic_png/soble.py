import cv2
import pdb
import numpy as np
from matplotlib import pyplot as plt

img0 = cv2.imread('synoptic_GONG_20100714.png',0)
img  = img0*img0

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=7)
sobel_xy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=7)

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.figure(2)
plt.imshow(sobel_xy,cmap = 'gray')
plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

plt.show()
