# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    frame = imutils.resize(frame, width=400)
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            print(person)
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
            for feature in keypoints.keys():
                cv2.putText(frame,'({}, {})'.format(keypoints[feature][0], keypoints[feature][1]),\
                    (keypoints[feature][0], keypoints[feature][1]+5), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,155,255) )
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()