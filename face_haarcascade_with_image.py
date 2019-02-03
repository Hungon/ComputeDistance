import numpy as np
import cv2
import time
from datetime import datetime

MAX_INDEX_FOR_PHOTOS = 44

if __name__ == '__main__':
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    index = 35
    while True:
        image_name = 'sample_face_'+str(index)+'.jpg'
        img = cv2.imread('sample_images/'+image_name,cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s_time = int(round(time.time() * 1000))
        faces = face_cascade.detectMultiScale(gray, 1.01, 3)
        e_time = int(round(time.time() * 1000))
        print("index:"+str(index)+" face_cascade took : "+str(e_time-s_time)+" ms")
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            print("index:"+str(index)+" fx:"+str(x)+" fy:"+str(y)+" fw:"+str(w)+" fh:"+str(h))

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                print("index:"+str(index)+" ex:"+str(ex)+" ey:"+str(ey)+" ew:"+str(ew)+" eh:"+str(eh))

        cv2.imshow("image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        index += 1
        index %= MAX_INDEX_FOR_PHOTOS
        time.sleep(2.0)

    cv2.destroyAllWindows()
