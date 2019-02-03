# import the necessary packages
import numpy as np
import cv2
import time

def find_face(image):
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = image_gray[y:y+h, x:x+w]
        print("fx:"+str(x)+" fy:"+str(y)+" fw:"+str(w)+" fh:"+str(h))
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		# finds edges in an image using the Canny86 algorithm if needed.

        (cnts, _) = cv2.findContours(roi_gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key = cv2.contourArea)
	    # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

if __name__ == '__main__':

	cap = cv2.VideoCapture(0)
	# initialize the known distance from the camera to the object in centimerter,
	KNOWN_DISTANCE = 90.0
	# initialize the known object width in centimerter,
	KNOWN_WIDTH = 15.0
	# load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize
	# the focal length
	image = cv2.imread("images/sample_face_2.jpg")
	marker = find_face(image)
	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

	while True:
		ret, img = cap.read()
		if img.any():
			marker = find_face(img)
			if marker:
				centi = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
				cv2.putText(img, "%.2fcm" % centi,
					(img.shape[1] - 400, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
				print('distance to object: '+str(centi))
				cv2.imshow('img',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
