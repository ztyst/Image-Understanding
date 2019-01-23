import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import math
from sklearn.metrics.pairwise import euclidean_distances


#===================================Question 1========================##
def find_homography():
	A = np.zeros((8,9))
	paper = (210,297)
	p1 = (152,243)
	p2 = (237,240)
	p3 = (156,371)
	p4 = (234,386)


	A[0,:] = np.array([p1[0],p1[1],1,0,0,0,-p1[0],-p1[1],-1])
	A[1,:] = np.array([0,0,0,p1[0],p1[1],1,-p1[0],-p1[1],-1])
	A[2,:] = np.array([p2[0],p2[1],1,0,0,0,-210*p2[0],-210*p2[1],-210])
	A[3,:] = np.array([0,0,0,p2[0],p2[1],1,-p2[0],-p2[1],-1])
	A[4,:] = np.array([p3[0],p3[1],1,0,0,0,-p3[0],-p3[1],-1])
	A[5,:] = np.array([0,0,0,p3[0],p3[1],1,-297*p3[0],-297*p3[1],-297])
	A[6,:] = np.array([p4[0],p4[1],1,0,0,0,-210*p4[0],-210*p4[1],-210])
	A[7,:] = np.array([0,0,0,p4[0],p4[1],1,-297*p4[0],-297*p4[1],-297])

	eigvalues, eigvectors = np.linalg.eig(np.dot(A.T, A))
	min_idx = eigvalues.argmin()   
	h = eigvectors[:,min_idx].reshape(3,3)
	return h


def q1():
	h = find_homography()
	print h
	# template = cv2.imread('paper2.jpg')
	# width,height = template.shape[:2]
	# paste = cv2.warpPerspective(template,h,(width,height))
	# cv2.imwrite('q1.png',paste)

	p1 = (92,109)
	p2 = (354,54)
	p3 = (110,547)
	p4 = (341,648)

	def findPoint(h,p):
		x = p[0]
		y = p[1]
		x_p = (h[0,0]*x+h[0,1]*y+h[0,2])/(h[2,0]*x+h[2,1]*y+h[2,2])
		y_p = (h[1,0]*x+h[1,1]*y+h[1,2])/(h[2,0]*x+h[2,1]*y+h[2,2])
		return (x_p, y_p)

	p1p = findPoint(h,p1)
	print p1p
	p2p = findPoint(h,p2)
	print p2p
	p3p = findPoint(h,p3)
	x_length = math.sqrt((p2p[0] - p1p[0])**2 + (p2p[1] - p1p[1])**2)
	y_length = math.sqrt((p3p[0] - p1p[0])**2 + (p3p[1] - p1p[1])**2)
	print x_length
	print y_length








#===================================Question 2========================##
# q2 helper
def readToGray(path):
		img = cv2.imread(path)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return img,gray

# q2 helper
def getKeypointsAndFeatures(img,img_gray):
	sift = cv2.xfeatures2d.SIFT_create()
	kp,des = sift.detectAndCompute(img_gray,None)
	return kp,des


# q2 heper
def featureMatching1(thres,k,des,des_t,kp,kp_t):
	match_counter = 0
	P = []
	P_prime = []
	result_map = {}
	for i in range(des_t.shape[0]):
		dist = euclidean_distances([des_t[i]],des)
		closest = np.amin(dist)
		closest_idx = np.argmin(dist)
		dist = np.delete(dist,closest_idx)
		sec_closest = np.amin(dist)
		ratio = closest/sec_closest
		if ratio < thres:
			match_counter += 1
			result_map[closest] = (kp[closest_idx],kp_t[i])
	
	# get top k correspondence
	keylist = result_map.keys()
	keylist = sorted(keylist)[:k]

	# compute P and P_prime
	for key in keylist:
		row1 = [result_map.get(key)[1].pt[0],result_map.get(key)[1].pt[1],0,0,1,0]
		row2 = [0,0,result_map.get(key)[1].pt[0],result_map.get(key)[1].pt[1],0,1]
		P.append(row1)
		P.append(row2)
		P_prime.append(result_map.get(key)[0].pt)
	P = np.array(P)
	P_prime = np.array(P_prime).flatten()

	return match_counter,P,P_prime

# q2 heper modified from assignment 2
def featureMatching(thres,k,des,des_t,kp,kp_t):
	match_counter = 0
	P = []
	P_prime = []
	result_map = {}
	for i in range(des_t.shape[0]):
		dist = euclidean_distances([des_t[i]],des)
		closest = np.amin(dist)
		closest_idx = np.argmin(dist)
		dist = np.delete(dist,closest_idx)
		sec_closest = np.amin(dist)
		ratio = closest/sec_closest
		if ratio < thres:
			match_counter += 1
			result_map[closest] = (kp[closest_idx],kp_t[i])
	
	# get random k samples
	keylist = result_map.keys()
	new_keylist = random.sample(keylist,k)

	# compute P and P_prime
	for key in new_keylist:
		row1 = [result_map.get(key)[1].pt[0],result_map.get(key)[1].pt[1],0,0,1,0]
		row2 = [0,0,result_map.get(key)[1].pt[0],result_map.get(key)[1].pt[1],0,1]
		P.append(row1)
		P.append(row2)
		P_prime.append(result_map.get(key)[0].pt)
	P = np.array(P)
	P_prime = np.array(P_prime).flatten()

	return match_counter,P,P_prime,result_map


def q2b():
	def computeS(P,k,p):
		return np.divide(np.log(1-P),np.log(1-np.power(p,k)))

	#affine (assume p=0.7)
	print computeS(0.99,3,0.95)
	print computeS(0.99,3,0.82)
	print computeS(0.99,3,0.11)
	#projective
	print computeS(0.99,4,0.95)
	print computeS(0.99,4,0.82)
	print computeS(0.99,4,0.11)


def q2c(source_img,template_img,output):
	#read image into grayscale
	img,img_gray = readToGray(source_img)
	template,template_gray = readToGray(template_img)

	# get keypoints and descriptors
	kp,des = getKeypointsAndFeatures(img,img_gray)
	kp_t,des_t = getKeypointsAndFeatures(template,template_gray)

	def computeA(P,P_prime):
		P_inverse = np.dot(np.linalg.pinv(np.dot(P.T,P)),P.T)
		return np.dot(P_inverse,P_prime)

	# find Affine matrix
	matching,P,P_prime = featureMatching1(0.8,3,des,des_t,kp,kp_t)
	A = computeA(P,P_prime)

	#RANSAC setup
	max_inliers = 0
	M = np.zeros((2,3))
	opt_P =np.zeros((6,6))
	opt_P_prime =np.zeros((6,))

	# From Part 2 b) 10 iterations should be good
	for i in range(10):
		# get random 3 samples and compute A
		matching,P,P_prime,result_map = featureMatching(0.8,3,des,des_t,kp,kp_t)
		A = computeA(P,P_prime)

		# using Affine matrix to map the points in img1 back to the img2
		X = []
		X_p_target = []
		for key in result_map.keys():
			row1 = [result_map.get(key)[1].pt[0],result_map.get(key)[1].pt[1],0,0,1,0]
			row2 = [0,0,result_map.get(key)[1].pt[0],result_map.get(key)[1].pt[1],0,1]
			X.append(row1)
			X.append(row2)
			X_p_target.append(result_map.get(key)[0].pt)
		X = np.array(X)

		X_p_target = np.array(X_p_target)
		X_p_result = np.dot(X,A)

		# compute the inliers whose euclidean distance is less than 3
		it = iter(X_p_result)
		X_p_result = np.array(zip(it,it))
		distance = euclidean_distances(X_p_target,X_p_result)
		inliers = distance[np.where(distance < 3.0)]

		# find A that gave the most inliers
		if inliers.shape[0] > max_inliers:
			max_inliers = inliers.shape[0]
			M[0,0] = A[0]
			M[0,1] = A[1]
			M[0,2] = A[4]
			M[1,0] = A[2]
			M[1,1] = A[3]
			M[1,2] = A[5]
			opt_P = P
			opt_P_prime = P_prime

	height,width = img.shape[:2]

	paste = cv2.warpAffine(template,M,(width,height))
	result_img = np.where(paste != [0,0,0] ,1, 0)*paste + np.where(paste == [0,0,0],1,0)*img
	cv2.imwrite("./result/"+output,result_img)

def q2d(source_img,template_img,output,output2):
	#read image into grayscale
	img,img_gray = readToGray(source_img)
	template,template_gray = readToGray(template_img)

	# get keypoints and descriptors
	kp,des = getKeypointsAndFeatures(img,img_gray)
	kp_t,des_t = getKeypointsAndFeatures(template,template_gray)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des_t,des)
	   
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False) 
	# Remove not so good matches
	numGoodMatches = int(len(matches) * 0.15)
	matches = matches[:numGoodMatches]
	img3 = cv2.drawMatches(template,kp_t,img,kp,matches, None,flags=2)
	cv2.imwrite('./result/'+output2,img3)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp_t[match.queryIdx].pt
		points2[i, :] = kp[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)

	# Use homography
	height, width, channels = img.shape
	im1Reg = cv2.warpPerspective(template, h, (width, height))
	result_img = np.where(im1Reg != [0,0,0] ,1, 0)*im1Reg + np.where(im1Reg == [0,0,0],1,0)*img

	cv2.imwrite('./result/'+output, result_img)


def q2e(source_img,template_img,template_2,output):
	#read image into grayscale
	img,img_gray = readToGray(source_img)
	template,template_gray = readToGray(template_img)
	template2,template_gray2 = readToGray(template_2)

	# get keypoints and descriptors
	kp,des = getKeypointsAndFeatures(img,img_gray)
	kp_t,des_t = getKeypointsAndFeatures(template,template_gray)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des_t,des)
	   
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False) 
	# Remove not so good matches
	numGoodMatches = int(len(matches) * 0.15)
	matches = matches[:numGoodMatches]

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp_t[match.queryIdx].pt
		points2[i, :] = kp[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)

	# Use homography
	height, width, channels = img.shape
	im1Reg = cv2.warpPerspective(template2, h, (width, height))
	result_img = np.where(im1Reg != [0,0,0] ,1, 0)*im1Reg + np.where(im1Reg == [0,0,0],1,0)*img

	cv2.imwrite('./result/'+output, result_img)

#===================================Question 3========================##


def computeK():
	p1 = (240,-362,1)
	p2 = (-478,506,1)
	p3 = (752,486,1)

	def computeA_row(pA,pB):
		x1,y1,z1 = pA
		x2,y2,z2 = pB
		row = [x1*x2+y1*y2,x1*z2+x2*z1,y1*z2+y2*z1,z1*z2]
		return row
	# compute A
	A = np.array([computeA_row(p1,p2),computeA_row(p2,p3),computeA_row(p3,p1)])
	r = np.linalg.svd(A)[2][-1]
	# compute matrix W
	w = np.array([[r[0],0,r[1]],[0,r[0],r[2]],[r[1],r[2],r[3]]],dtype = "float")
	K = np.linalg.inv(np.linalg.cholesky(w).T)
	#normalize the K matrix
	K = K/K[2][2]
	print K


##=============================== Question 4 ===================================================#
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
	return mid_img

# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):
	# get keypoints and descriptors
	sift = cv2.xfeatures2d.SIFT_create()

	# Extract keypoints and descriptors
	kp, des = sift.detectAndCompute(img2, None)
	kp_t, des_t = sift.detectAndCompute(img1, None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des_t,des)
	   
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False) 
	# Remove not so good matches
	numGoodMatches = int(len(matches) * 0.15)
	matches = matches[:numGoodMatches]

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp_t[match.queryIdx].pt
		points2[i, :] = kp[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)
	return h

def q4():
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

	# flip the images 
	img5 = np.flip(img5,1)
	img6 = np.flip(img6,1)
	img7 = np.flip(img7,1)
	img8 = np.flip(img8,1)
	img9 = np.flip(img9,1)
	img_list_right = [img9,img8,img7,img6,img5]

	#flip back the outcome image on rigth side
	result_image_right = np.flip(get_stitched_image(img_list_right),1)
	result_image_left = get_stitched_image(img_list_left)
	result_image = get_stitched_image([result_image_left,result_image_right])

	cv2.imwrite('./result/mypano.jpg', result_image)

#===================================Question 4========================##

if __name__ == "__main__":
	#q2c('./A3/data/in4.jpg','./A3/data/book.jpg','affine_2.jpg')
	#q2d('./A3/data/in5.jpg','./A3/data/book.jpg','homopgraphy5.jpg','homopgraphy5_matches.jpg')
	#q1()
	#computeK()
	#q2e('./A3/data/in1.jpg','./A3/data/book.jpg','./A3/data/book2.jpg','homopgraphy1_e.jpg')	
	#q2b()
	q4()