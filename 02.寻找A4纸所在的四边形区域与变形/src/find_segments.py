import numpy as np
import cv2


gray = cv2.imread('./raw_numbers_img.jpg', cv2.IMREAD_GRAYSCALE)


# gray_blur = cv2.GaussianBlur(gray,(3,3),0) 
gray_blur = cv2.medianBlur(gray,5) 
edges = cv2.Canny(gray_blur, 50, 150, apertureSize = 3)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  
edges_dilate = cv2.dilate(edges, kernel, iterations=1)
lines = cv2.HoughLinesP(edges_dilate, 10, np.pi/180, 1000) #这里对最后一个参数使用了经验型的值

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))  
edges = cv2.dilate(edges, kernel, iterations=3)

canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for line in lines:
    
    (x1, y1, x2, y2) = line[0]
    cv2.line(canvas, (x1, y1), (x2, y2), (0,0,255), 2)  

cv2.namedWindow('Canny', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.namedWindow('Result', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.imshow('Canny', edges_dilate )  
cv2.imshow('Result', canvas)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  

