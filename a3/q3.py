import cv2
import numpy as np

img = cv2.imread('./A3/data/1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,100,200)
result = np.zeros((2000,2000))
lines = cv2.HoughLines(edges,1,np.pi/180,200)
result[750:1250,750:1250] = gray


cv2.imwrite('warp1.jpg',result)