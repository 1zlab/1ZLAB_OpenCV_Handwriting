'''
A4纸投影
'''

import cv2
import numpy as np


gray = cv2.imread('raw_numbers_img.jpg', cv2.IMREAD_GRAYSCALE)

# 因为之前膨胀了很多次，所以四边形区域需要向内收缩而且本身就有白色边缘
margin=40
pts1 = np.float32([[921+margin, 632+margin], [659+margin, 2695-margin], [3795-margin, 2630-margin], [3362-margin, 856+margin]])

pts2 = np.float32([[0,0], [0, 1000], [1400, 1000], [1400, 0]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(gray,M,(1400,1000))

cv2.imwrite('perpective-number.png', dst)