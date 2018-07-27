import numpy as np
import cv2

gray = cv2.imread('./raw_numbers_img.jpg', cv2.IMREAD_GRAYSCALE)


# gray_blur = cv2.GaussianBlur(gray,(3,3),0) 
gray = cv2.medianBlur(gray,5) 
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

cv2.imwrite('numbers_edge_raw.png', edges)