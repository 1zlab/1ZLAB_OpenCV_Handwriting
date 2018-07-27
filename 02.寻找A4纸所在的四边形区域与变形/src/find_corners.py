# -*- coding: utf-8 -*-
import cv2

cv2.namedWindow('original', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.namedWindow('SIFT_features', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)


# 读取图像
img = cv2.imread('raw_numbers_img.jpg')
cv2.imshow('original',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.medianBlur(gray,5) 
edges = cv2.Canny(gray_blur, 50, 150, apertureSize = 3)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  
edges = cv2.dilate(edges, kernel, iterations=1)


#cv2.waitKey()

# 检测特征点
sift = cv2.xfeatures2d.SIFT_create() # 调用SURF
# 获取关键点
keypoints = sift.detect(edges_dilate)


canvas = cv2.cvtColor(edges_dilate, cv2.COLOR_GRAY2BGR)

# 显示特征点
for k in keypoints:
    cv2.circle(canvas,(int(k.pt[0]),int(k.pt[1])),10,(0,255,0),-1)

cv2.imshow('SIFT_features',canvas)
cv2.waitKey()
cv2.destroyAllWindows()
