import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import scipy as sp
def q1a():

	def harris(det,trace):
		R = det - 0.05 * np.multiply(trace,trace)
		R = cv2.normalize(R,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
		cv2.imwrite("harris.png",R)
		return R

	def brown(det,trace):
		B = np.divide(det, trace, out=np.zeros_like(det), where=trace!=0)
		B = cv2.normalize(B,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)
		cv2.imwrite("brown.png",B)
		return B

	# q1b
	def nms(B,r,threshold):
		c,l = B.shape
		thres = np.amax(B)*threshold

		# Create a circular mask
		y,x = np.ogrid[-r:r+1,-r:r+1]
		circular_mask = x**2+y**2 <= r**2
		circular_mask = 1*circular_mask.astype(float)

		# mark the keypoints in the origin image
		img = cv2.imread('building.jpg')

		# display keypoints
		result = np.zeros((c,l))

		for i in range(r,c-r):
			for j in range(r,l-r):
				patch = B[i-r:i+r+1,j-r:j+r+1]
				patch = np.multiply(circular_mask,patch)

				if B[i,j] > thres and B[i,j] == patch.max():
					result[i,j]=255
					cv2.circle(img,(j,i),5,(255,0,0))

		cv2.imwrite('nms2.jpeg',result)
		cv2.imwrite('nms_img2.jpeg',img)


	img = cv2.imread('building.jpg')
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),7)
	Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
	Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
	IxIy = np.multiply(Ix, Iy)
	Ix2 = np.multiply(Ix, Ix)
	Iy2 = np.multiply(Iy, Iy)
	Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
	Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
	IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10) 
	det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
	trace = Ix2_blur + Iy2_blur
	R = harris(det,trace)
	B = brown(det,trace)
	nms(B,50,0.001)


from scipy import ndimage
def q1c(sigma_max):
	img = cv2.imread('synthetic.png')
	#img = img.astype(np.float32)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	LoG = np.zeros((gray.shape[0],gray.shape[1],sigma_max))
	s = np.power(1.5,np.arange(sigma_max))
	print s
	for i in range(len(s)):
		
		LoG[:,:,i] = s[i]**2*ndimage.gaussian_laplace(gray,sigma=s[i])

	for z in range(1,LoG.shape[2]-1):
		thres = np.amax(LoG[:,:,z])*0.9
		for y in range(2,LoG.shape[1]-1):
			for x in range(2,LoG.shape[0]-1):

				if abs(LoG[x,y,z]) >= thres:
					top_max = np.max(LoG[x-1:x+2,y-1:y+2,z+1])
					current_max = np.max(LoG[x-1:x+2,y-1:y+2,z])
					bottum_max = np.max(LoG[x-1:x+2,y-1:y+2,z-1])
					if LoG[x,y,z] > top_max and LoG[x,y,z] > bottum_max and LoG[x,y,z] == current_max:
						cv2.circle(img,(y,x),5,(255,0,0),1)
				if abs(LoG[x,y,z]) <= 20:
					top_min = np.min(LoG[x-1:x+2,y-1:y+2,z+1])
					current_min = np.min(LoG[x-1:x+2,y-1:y+2,z])
					bottum_min = np.min(LoG[x-1:x+2,y-1:y+2,z-1])
					if LoG[x,y,z] < top_min and LoG[x,y,z] < bottum_min and LoG[x,y,z] == current_min:
						cv2.circle(img,(y,x),3,(0,255,0))


	cv2.imwrite('q1c.png',img)



def q1d():
	img = cv2.imread('synthetic.png',0)
	surf = cv2.xfeatures2d.SURF_create(2000)
	kp,des = surf.detectAndCompute(img,None)
	img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
	cv2.imwrite('q1d.png',img2)


#=============================Question 2==============================#

# q2 helper
def readToGray(path):
		img = cv2.imread(path)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		return img,gray

# q2 helper
def getKeypointsAndFeatures(img,img_gray):
	sift = cv2.xfeatures2d.SIFT_create()
	kp,des = sift.detectAndCompute(img_gray,None)
	output = cv2.drawKeypoints(img_gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return output,kp,des

# q2 heper
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


#q2 helper
def drawAffine(P, P_prime,inputFile,template,outputFile):
	#Compute affine matrix
	def computeA(P,P_prime):
		P_inverse = np.dot(np.linalg.pinv(np.dot(P.T,P)),P.T)
		return np.dot(P_inverse,P_prime)

	img2,img_gray2 = readToGray(inputFile)
	template,template_gray = readToGray(template)
	# Create P matrix from reference image
	ref_row, ref_col = template_gray.shape
	P_new = np.array([[1,1,0,0,1,0],[0,0,1,1,0,1],\
		[1,ref_row,0,0,1,0],[0,0,1,ref_row,0,1],[ref_col,1,0,0,1,0],[0,0,ref_col,1,0,1],\
		[ref_col,ref_row,0,0,1,0],[0,0,ref_col,ref_row,0,1]])

	# Generate P_prime from Affine
	print(computeA(P,P_prime))
	P_prime_new = np.dot(P_new,computeA(P,P_prime)).astype(int)

	#Draw lines on source image
	cv2.line(img2,(P_prime_new[0],P_prime_new[1]),(P_prime_new[2],P_prime_new[3]),(255,0,0),5)
	cv2.line(img2,(P_prime_new[4],P_prime_new[5]),(P_prime_new[6],P_prime_new[7]),(255,0,0),5)
	cv2.line(img2,(P_prime_new[4],P_prime_new[5]),(P_prime_new[0],P_prime_new[1]),(255,0,0),5)
	cv2.line(img2,(P_prime_new[6],P_prime_new[7]),(P_prime_new[2],P_prime_new[3]),(255,0,0),5)
	cv2.imwrite(outputFile,img2)



def q2a():
	#read image into grayscale
	img,img_gray = readToGray('findBook.png')
	template,template_gray = readToGray('book.jpeg')

	# get keypoints and descriptors
	output,kp,des = getKeypointsAndFeatures(img,img_gray)
	output_template,kp_t,des_t = getKeypointsAndFeatures(template,template_gray)

	#write keypoints into images
	cv2.imwrite("sift_template.png",output_template)
	cv2.imwrite("sift.png",output)

	# Feature match: Get P and P_prime

	# matching4 = featureMatching(0.4,3,des,des_t,kp,kp_t)
	matching5,P5,P_prime5 = featureMatching(0.5,3,des,des_t,kp,kp_t)
	matching6,P6,P_prime6 = featureMatching(0.6,3,des,des_t,kp,kp_t)
	matching7,P7,P_prime7 = featureMatching(0.7,3,des,des_t,kp,kp_t)
	matching8,P8,P_prime8 = featureMatching(0.8,3,des,des_t,kp,kp_t)
	matching9,P9,P_prime9 = featureMatching(0.9,3,des,des_t,kp,kp_t)
	plt.plot([0.5,0.6,0.7,0.8,0.9],[matching5,matching6,matching7,matching8,matching9],'ro')
	plt.savefig('plot.png')

	drawAffine(P,P_prime,'findBook.png','book.jpeg','q2c3.png')




def q2e():
	weight = 2

	img,img_gray = readToGray('colourSearch.png')
	template,template_gray = readToGray('colourTemplate.png')

	# get keypoints and descriptors
	output,kp,des = getKeypointsAndFeatures(img,img_gray)
	output_template,kp_t,des_t = getKeypointsAndFeatures(template,template_gray)

	#Padd Features with RGB Values
	des = np.append(des, np.zeros((des.shape[0],3)), axis=1)

	des_t = np.append(des_t, np.zeros((des_t.shape[0],3)), axis=1)

	for i in range(len(kp)):
		x, y  = kp[i].pt
		colors = img[int(y),int(x),:]
		des[i, -3:] = colors*weight

	for i in range(len(kp_t)):
		x, y  = kp_t[i].pt
		colors = template[int(y),int(x),:]
		des_t[i, -3:] = colors*weight

	# Feature match: Get P and P_prime
	matching,P,P_prime = featureMatching(0.8,3,des,des_t,kp,kp_t)
	drawAffine(P,P_prime,'colourSearch.png','colourTemplate.png','q2e.png')



#=============================Question 3==============================#


def q3():
	def computeS(P,k,p):
		return np.divide(np.log(1-P),np.log(1-np.power(p,k)))

	k1 = np.arange(1,20)
	plt.plot(k1,computeS(0.99,k1,0.7))
	plt.savefig("q3a.png")
	plt.clf()

	k2 = 5
	p2 = np.arange(0.1,0.5,0.01)
	plt.plot(p2,computeS(0.99,5,p2))
	plt.savefig("q3b.png")

	print(computeS(0.99,5,0.2))




if __name__ == "__main__":
	q1d()
	q1c(sigma_max=5)
	q1a()
	q2a()
	q3()
	q2e()