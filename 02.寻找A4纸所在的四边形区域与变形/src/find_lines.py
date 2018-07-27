import numpy as np
import cv2

gray = cv2.imread('./raw_numbers_img.jpg', cv2.IMREAD_GRAYSCALE)


# gray_blur = cv2.GaussianBlur(gray,(3,3),0) 
gray = cv2.medianBlur(gray,5) 
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))  
edges = cv2.dilate(edges, kernel, iterations=10)


# 寻找轮廓
bimg, contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))

# 定义一个二值图
edges_filter = np.zeros_like(gray)
# 用白线绘制边缘
cv2.drawContours(edges_filter, [cnt], 0, (255), 3)
# cv2.imwrite('number-edge-filter.png', edges_filter)
# 霍夫变换，寻找直线
lines = cv2.HoughLines(edges_filter, 6, np.pi/90, 500) #这里对最后一个参数使用了经验型的值  

# 定义画布
canvas = cv2.cvtColor(edges_filter, cv2.COLOR_GRAY2BGR)
# 获取画布尺寸
height,width,_ = canvas.shape

for line in lines:
    rho = line[0][0] #第一个元素是距离rho  
    theta= line[0][1] #第二个元素是角度theta  
    print(rho)
    print(theta)  
    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线  
                #该直线与第一行的交点  
        pt1 = (int(rho/np.cos(theta)),0)  
        #该直线与最后一行的交点  
        pt2 = (int((rho-height*np.sin(theta))/np.cos(theta)),height)  
        #绘制一条绿线  
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 4)  
    else: #水平直线  
        # 该直线与第一列的交点  
        pt1 = (0,int(rho/np.sin(theta)))  
        #该直线与最后一列的交点  
        pt2 = (width, int((rho-width*np.cos(theta))/np.sin(theta)))  
        #绘制一条直线  
        cv2.line(canvas, pt1, pt2, (0,0,255), 4)  


cv2.namedWindow('Canny', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.namedWindow('Result', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
cv2.imshow('Canny', edges_filter )  
cv2.imshow('Result', canvas)

cv2.imwrite('numbers_contours_find_lines.png', canvas)
cv2.waitKey(0)  
cv2.destroyAllWindows()  

