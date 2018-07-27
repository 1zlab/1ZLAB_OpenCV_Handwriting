import cv2
import numpy as np


# gray = cv2.imread('numbers_A4.png', cv2.IMREAD_GRAYSCALE)
gray = cv2.imread('numbers_gray_erode.png', cv2.IMREAD_GRAYSCALE)


gray = cv2.medianBlur(gray,3)


numbers_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imwrite('numbers_bin_gaussian.png', numbers_bin)

numbers_bin = cv2.medianBlur(numbers_bin,3)
cv2.imwrite('numbers_bin_gaussian_filter_by_meidan.png', numbers_bin)