import numpy as np
import cv2
import pickle
import utils

# Load face haarcascade classifier in opencv
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml") # after get the trained modle

# Load text labels
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    # inverting dict to show names according to number
    labels = {val:key for key,val in og_labels.items()}

cap = cv2.VideoCapture(0)
# width = cap.get(3)
# height = cap.get(4)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = utils.rescale_frame(frame, percent = 50)
    # Detect faces by using the cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # factors according to documentations
    for (x, y, w, h) in faces: # x, y starting from top left coordinate
        # print x, y, w, h # print out detected face coordinates

        # Range out the region of interest (face!)
        roi_gray = gray[y:y+h, x:x+w]

        # # Save the ROI of detected face
        # img_item = "my-face.png"
        # img_item2 = "my-face2.png"
        # cv2.imwrite(img_item, roi_gray)

        # Recognition of faces
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 85:
            print id_
            print labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            colour = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, colour, stroke,cv2.LINE_AA)

        # Draw rectangle of ROI
        roi_colour = frame[y:y+h, x:x+w]
        colour = (255, 0, 0)
        stroke = 2 # thickness of the line
        end_x = x + w
        end_y = y + h
        cv2.rectangle(frame, (x, y), (end_x, end_y), colour, stroke) # here is the rectangle coor rather than h&w



    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()