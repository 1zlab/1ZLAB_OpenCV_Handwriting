import numpy as np
import cv2


gray = cv2.imread('./raw_numbers_img.jpg', cv2.IMREAD_GRAYSCALE)


# gray_blur = cv2.GaussianBlur(gray,(3,3),0) 
gray = cv2.medianBlur(gray,5) 
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  
edges = cv2.dilate(edges, kernel, iterations=1)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))  
edges = cv2.dilate(edges, kernel, iterations=10)


# 寻找轮廓
bimg, contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))

canvas = np.zeros((gray.shape[0], gray.shape[1],3))
cv2.drawContours(canvas, [cnt], 0, (0,255,0), 3)

cv2.imwrite('number-edge.png',edges)
cv2.imwrite('number-contours.png', canvas)
