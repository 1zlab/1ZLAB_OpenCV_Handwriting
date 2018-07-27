'''
多边形近似提取A4纸的多边形点集
（四边形近似）
'''
import numpy as np
import cv2


# 读入图片
gray = cv2.imread('./raw_numbers_img.jpg', cv2.IMREAD_GRAYSCALE)
# 中值滤波 过滤噪声，保留边缘信息
gray = cv2.medianBlur(gray,5) 
# Canny算子求得图像边缘
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

# 定义一个9×9的十字形状的结构元
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
# 重复膨胀 迭代10次
edges = cv2.dilate(edges, kernel, iterations=10)


# 寻找轮廓
bimg, contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 获取面积最大的contour
cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))

edges_filter = np.zeros_like(gray)
cv2.drawContours(edges_filter, [cnt], 0, (255), 3)
cv2.imwrite('number-edge-filter.png', edges_filter)

# 多变形近似
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)


canvas = cv2.cvtColor(edges_filter, cv2.COLOR_GRAY2BGR)
height,width,_ = canvas.shape

# print(approx)
cv2.drawContours(canvas, [approx], 0, (0, 255, 0), 10)
for point in approx:
    # 绘制角点
    cv2.circle(canvas,tuple(point[0]), 40, (0,255, 255), -1)

cv2.namedWindow('Canny', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.namedWindow('Result', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.imshow('Canny', edges_filter )  
cv2.imshow('Result', canvas)

# cv2.imwrite('countour-approximation-corner-point.png', canvas)
cv2.waitKey(0)  
cv2.destroyAllWindows()  

