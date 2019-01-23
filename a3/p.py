import cv2

stitcher = cv2.createStitcher(False)
img1 = cv2.imread('./A3/data/landscape_1.jpg')
img2 = cv2.imread('./A3/data/landscape_2.jpg')
img3 = cv2.imread('./A3/data/landscape_3.jpg')
img4 = cv2.imread('./A3/data/landscape_4.jpg')
img5 = cv2.imread('./A3/data/landscape_5.jpg')
img6 = cv2.imread('./A3/data/landscape_6.jpg')
img7 = cv2.imread('./A3/data/landscape_7.jpg')
img8 = cv2.imread('./A3/data/landscape_8.jpg')
img9 = cv2.imread('./A3/data/landscape_9.jpg')
result = stitcher.stitch((img1,img2,img3,img4,img5,img6,img7,img8,img9))

cv2.imwrite("result.jpg", result[1])