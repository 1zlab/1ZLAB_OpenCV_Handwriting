
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 载入图片
img = cv2.imread('numbers_A4.png')


lowerb = (0,0,116)
upperb = (255,255,255)
# 根据hsv阈值 进行二值化
back_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowerb, upperb)

cv2.imwrite('number_back_mask_by_hsv_threshold.png', back_mask)

# 形态学操作， 圆形核腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
back_mask = cv2.erode(back_mask, kernel, iterations=1)
# 反色 变为数字的掩模
num_mask = cv2.bitwise_not(back_mask)
# 中值滤波
num_mask = cv2.medianBlur(num_mask,3) 
cv2.imwrite('number_mask_filter_by_median.png', num_mask)

# 寻找轮廓
bimg, contours, hier = cv2.findContours(num_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 声明画布 拷贝自img
canvas = cv2.cvtColor(num_mask, cv2.COLOR_GRAY2BGR)

def getStandardDigit(img):
    '''
        返回标准的数字矩阵
    '''
    STD_WIDTH = 32 # 标准宽度
    STD_HEIGHT = 64

    height,width = img.shape
    
    # 判断是否存在长条的1
    new_width = int(width * STD_HEIGHT / height)
    if new_width > STD_WIDTH:
        new_width = STD_WIDTH
    # 以高度为准进行缩放
    resized_num = cv2.resize(img, (new_width,STD_HEIGHT), interpolation = cv2.INTER_NEAREST)
    # 新建画布
    canvas = np.zeros((STD_HEIGHT, STD_WIDTH))
    x = int((STD_WIDTH - new_width) / 2) 
    canvas[:,x:x+new_width] = resized_num
    
    return canvas

minWidth = 5 # 最小宽度
minHeight = 20 # 最小高度

base = 1000 # 计数编号
imgIdx = base # 当前图片的编号

# 检索满足条件的区域
for cidx,cnt in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(cnt)
    if w < minWidth or h < minHeight:
        # 如果不满足条件就过滤掉
        continue
    # 获取ROI图片
    digit = num_mask[y:y+h, x:x+w]
    digit = getStandardDigit(digit)
    cv2.imwrite('./digits_bin/{}.png'.format(imgIdx), digit)
    imgIdx+=1

    # 原图绘制圆形
    cv2.rectangle(canvas, pt1=(x, y), pt2=(x+w, y+h),color=(0, 255, 255), thickness=2)
    
cv2.imwrite('number_mask_mark_rect.png', canvas)