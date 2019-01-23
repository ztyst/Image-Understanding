def detection(object,image1,image2):
	dection_dict = detect object using tensorflow
    filtered_dict = filterDectionWithTopScore(dection_dict)
    depth = getDepth(camera_calibration,image1,image2)
    (X,Y,Z) = cacluate3DLocation(filtered_dict,depth)
    return (X,Y,Z), filtered_dict["score"]

#go to ball and pickup
def pickUp(X,Y,Z):
	faceToLocation(X,Y,Z)
	moveToLocation(X,Y,Z)
    grabObject(X,Y,Z)

#return to home and dance
def returnHome(X,Y,Z):
	faceToLocation(X,Y,Z)
	moveToLocation(X,Y,Z)
	victoryDanceAtLocation(X,Y,Z)

# take 2 pano photo
def takePanoPhoto():
	for every 20 degree:
        rotate robot
        len1 capture image1, append to image1_list
        len2 capture image2, append to image2_list
    image1_pano = getPanorama(image1_list)
    image2_pano = getPanorama(image2_list)

# check if the ball is in court
def checkInCourt(X_sideline,Y_sideline,Z_sideline,X_ball,Y_ball,Z_ball):
	distance_sideline = norm(X_sideline,Y_sideline,Z_sideline)
	distance_ball = norm(X_ball,Y_ball,Z_ball)
	return distance_ball >= distance_sideline


def main():
	while 1:
		image1_pano, image2_pano = takePanoPhoto()

		sideline1, sideline2 = cannyEdgeDetection(image1_pano), cannyEdgeDetection(image2_pano)
		X_sideline,Y_sideline = getXY(sideline1,sideline2)
		Z_sideline = getDepth(camera_calibration,sideline1,sideline2)

		(X_ball,Y_ball,Z_ball),score_ball = detection("ball",image1_pano,image2_pano)
		(X_obstacle,Y_obstacle,Z_obstacle), score_obstacle= detection("obstacle",image1_pano,image2_pano)
		(X_player,Y_player,Z_player), score_player = detection("player",image1_pano,image2_pano)

		if score_ball >= 0.5:

			if checkInCourt(X_sideline,Y_sideline,Z_sideline,X_ball,Y_ball,Z_ball) and score_player >= 0.5:
				continue

			if (X_obstacle,Y_obstacle,Z_obstacle) in same direction (X_ball,Y_ball,Z_ball) and score_obstacle >= 0.5:
				continue


			pickup(X_ball,Y_ball,Z_ball)

			image1_2_pano, image2_2_pano= takePanoPhoto()
			(X_home,Y_home,Z_home) = detection("player",image1_2_pano, image2_2_pano)

			returnHome(X_home,Y_home,Z_home)

