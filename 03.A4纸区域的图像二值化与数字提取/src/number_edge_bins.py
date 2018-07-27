import numpy as np
import cv2
gray = cv2.imread('numbers_A4.png', cv2.IMREAD_GRAYSCALE)


gray = cv2.medianBlur(gray,5) 
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
cv2.imwrite('numbers_edge.png', edges)


kernel = np.ones((5,5))
#opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,iterations=2)
# cv2.imwrite('numbers_edge_after_opening.png', opening)
dilated = cv2.dilate(edges, kernel, iterations=1)
cv2.imwrite('numbers_edge_after_dilated.png', dilated)