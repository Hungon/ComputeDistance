# import the necessary packages
import numpy as np
import cv2
import time

def find_markers_with_reducing_noise(image):
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    markers = []
    for (x,y,w,h) in faces:
        roi_gray = image_gray[y:y+h, x:x+w]
        print("fx:"+str(x)+" fy:"+str(y)+" fw:"+str(w)+" fh:"+str(h))
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		# finds edges in an image using the Canny86 algorithm if needed.
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        # remove noise in image
        edged = cv2.Canny(roi_gray, 35, 125)
        if edged.any():
            (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key = cv2.contourArea)
    	    # compute the bounding box of the of the paper region and return it
            markers.append(cv2.minAreaRect(c))
    return markers

def find_markers_without_reducing_noise(image):
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    markers = []
    for (x,y,w,h) in faces:
        roi_gray = image_gray[y:y+h, x:x+w]
        print("fx:"+str(x)+" fy:"+str(y)+" fw:"+str(w)+" fh:"+str(h))
        array_of_pos.append([x,y])
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        if roi_gray.any():
            (cnts, _) = cv2.findContours(roi_gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key = cv2.contourArea)
    	    # compute the bounding box of the of the paper region and return it
            markers.append(cv2.minAreaRect(c))
    return markers

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    index = 0
    MAX_INDEX = 10000
    MAX_EPOC = 10
    MAX_LENGTH_OF_MEAN_DISTANCE = 20
    THRESHOLD_MOVEMENT_Y = 50.0
    array_of_distance = []
    array_of_pos = []
    mean_array_of_distance = []
    mean_array_of_pos = []
    current_epoc = 0
    s_time_at_epoc = 0
    e_time_at_epoc = 0
	# initialize the known distance from the camera to the object in centimerter,
    KNOWN_DISTANCE = 85.0
	# initialize the known object width in centimerter,
    KNOWN_WIDTH = 15.0
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize
	# the focal length
    image = cv2.imread("images/sample_face_2.jpg")
    marker = find_markers_without_reducing_noise(image)[0]
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    while True:
        ret, img = cap.read()
        if img.any():
            s_time = int(round(time.time() * 1000))
            markers = find_markers_without_reducing_noise(img)
            e_time = int(round(time.time() * 1000))
            # compute distance to object
            if markers:
                for m in markers:
                    per_w = m[1][0]
                    print("index: "+str(index)+" per width: "+str(per_w)+" getting markers took : "+str(e_time-s_time)+" ms")
                    centi = distance_to_camera(KNOWN_WIDTH, focalLength, per_w)
                    array_of_distance.append(centi)
                    if len(array_of_distance) >= MAX_LENGTH_OF_MEAN_DISTANCE:
                        mean_distance = np.mean(array_of_distance)
                        mean_pos = np.mean(array_of_pos, axis=0)
                        mean_array_of_pos.append({current_epoc: mean_pos})
                        mean_array_of_distance.append({current_epoc: mean_distance})
                        length_mean_array_of_pos = len(mean_array_of_pos)
                        length_mean_array_of_distance = len(mean_array_of_distance)
                        # compute diffrence betweeen current one and first one
                        if length_mean_array_of_pos > 1:
                            diff_pos = mean_array_of_pos[0].get(0, 0)-mean_array_of_pos[length_mean_array_of_pos-1].get(current_epoc,0)
                            print("diff pos: "+str(diff_pos))
                        if length_mean_array_of_distance > 1:
                            diff_dis = mean_array_of_distance[0].get(0, 0)-mean_array_of_distance[length_mean_array_of_distance-1].get(current_epoc,0)
                            print("diff dis: "+str(diff_dis))
                        array_of_distance = []
                        array_of_pos = []
                        current_epoc += 1
                        if current_epoc % MAX_EPOC == 0:
                            mean_array_of_pos = []
                            mean_array_of_distance = []
                            current_epoc = 0

                    cv2.putText(img, "%.2fcm" % centi, (img.shape[1] - 400, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                    print("index: "+str(index)+' distance to object: '+str(centi)+" cm")
                    index += 1
                    index %= MAX_INDEX
        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
