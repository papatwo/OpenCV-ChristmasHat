import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# width = cap.get(3)
# height = cap.get(4)

def make_480p():
	cap.set(3, 640)
	cap.set(4, 480)


def rescale_frame(frame, percent = 75):
	scale_percent = percent
	width = int(frame.shape[1] * scale_percent / 100)
	height = int(frame.shape[0] * scale_percent / 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

while(1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print frame.shape
    frame = rescale_frame(frame, percent = 50)
    print frame.shape

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    # cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('photo.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()