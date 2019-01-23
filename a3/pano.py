import sys
import cv2
import numpy as np

# Use the keypoints to stitch the images
def get_stitched_image(img_list):
	mid_img = img_list[0]
	for img in img_list[1:]:
		M =  get_sift_homography(mid_img, img)

		# Get width and height of input images	
		w1,h1 = img.shape[:2]
		w2,h2 = mid_img.shape[:2]

		transform_array = np.array([[1, 0, h2], 
									[0, 1, 0], 
									[0,0,1]])
		M = np.dot(transform_array,M)
		dist_corner = cv2.warpPerspective(mid_img,M,(h1+h2,w2))
		mask = 255*np.ones(img.shape,img.dtype)
		center = (h2+h1//2,w1//2)
		dist_corner[0:w2,h2+100:h1+h2] = img[:,100:]

		mid_img = dist_corner
		# img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
		# img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)
		# print M


		# # Get relative perspective of second image
		# img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

		# # Resulting dimensions
		# result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

		# # Getting images together
		# # Calculate dimensions of match points
		# [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
		# [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
		
		# # affine account for translation
		# transform_dist = [-x_min,-y_min]
		# # print transform_dist
		# transform_array = np.array([[1, 0, -x_min], 
		# 							[0, 1, -y_min], 
		# 							[0,0,1]]) 
		# # M = np.dot(transform_array,M)
		# # dist_corner = cv2.warpPerspective(mid_img,M,(h1+h2,w2))
		# # mask = 255*np.ones(img.shape,img.dtype)
		# # center = (h2+h1//2,w1//2)
		# # dist_corner[0:w2,h2+100:h1+h2] = img[:,100:]
		# # mid_img = cv2.seamlessClone(img,dist_corner,mask,center,cv2.NORMAL_CLONE)
		# result_img = cv2.warpPerspective(mid_img, transform_array.dot(M), 
		#  								(x_max-x_min, y_max-y_min))
		# print result_imag.shape
		# print img.shape
		# # result_img[transform_dist[1]:w1+transform_dist[1], 
		# # 			transform_dist[0]:h1+transform_dist[0]] = img
		# mid_img = result_img
	# Return the result
	return mid_img

# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):

	# Initialize SIFT 
	sift = cv2.xfeatures2d.SIFT_create()

	# Extract keypoints and descriptors
	k1, d1 = sift.detectAndCompute(img1, None)
	k2, d2 = sift.detectAndCompute(img2, None)

	# Bruteforce matcher on the descriptors
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(d1,d2, k=2)

	# Make sure that the matches are good
	verify_ratio = 0.8 
	verified_matches = []
	for m1,m2 in matches:
		# Add to array only if it's a good match
		if m1.distance < 0.8 * m2.distance:
			verified_matches.append(m1)

	# Mimnum number of matches
	min_matches = 8
	if len(verified_matches) > min_matches:
		
		# Array to store matching points
		img1_pts = []
		img2_pts = []

		# Add matching points to array
		for match in verified_matches:
			img1_pts.append(k1[match.queryIdx].pt)
			img2_pts.append(k2[match.trainIdx].pt)
		img1_pts = np.float32(img1_pts).reshape(-1,1,2)
		img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		
		# Compute homography matrix
		M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
		return M
	else:
		print 'Error: Not enough matches'
		exit()

# Equalize Histogram of Color Images
def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img

# Main function definition
def main():
	
	# Get input set of images
	img1 = cv2.imread('./A3/data/landscape_1.jpg')
	img2 = cv2.imread('./A3/data/landscape_2.jpg')
	img3 = cv2.imread('./A3/data/landscape_3.jpg')
	img4 = cv2.imread('./A3/data/landscape_4.jpg')
	img5 = cv2.imread('./A3/data/landscape_5.jpg')
	img6 = cv2.imread('./A3/data/landscape_6.jpg')
	img7 = cv2.imread('./A3/data/landscape_7.jpg')
	img8 = cv2.imread('./A3/data/landscape_8.jpg')
	img9 = cv2.imread('./A3/data/landscape_9.jpg')
	img_list_left = [img1,img2,img3,img4,img5]
	img5 = np.flip(img5,1)
	img6 = np.flip(img6,1)
	img7 = np.flip(img7,1)
	img8 = np.flip(img8,1)
	img9 = np.flip(img9,1)
	img_list_right = [img9,img8,img7,img6,img5]


	


	# Equalize histogram
	# img1 = equalize_histogram_color(img1)
	# img2 = equalize_histogram_color(img2)

	# Stitch the images together using homography matrix
	result_image_right = np.flip(get_stitched_image(img_list_right),1)
	result_image_left = get_stitched_image(img_list_left)
	result_image = get_stitched_image([result_image_left,result_image_right])

	# Write the result to the same directory

	cv2.imwrite('./result/mypano.jpg', result_image)


	# # Show the resulting image
	# cv2.imshow ('Result', result_image)
	# cv2.waitKey()

# Call main function
if __name__=='__main__':
	main()