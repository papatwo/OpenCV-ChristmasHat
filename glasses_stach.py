import numpy as np
import cv2
import utils


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/thirdparty_frontaleyes35x15.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/thirdparty_nose25x15.xml')

glasses = cv2.imread('images/glasses.png', -1)
mustache = cv2.imread('images/mustache.png', -1)




cap = cv2.VideoCapture(0)

''' Water mark section
img_path = 'images/logo/cfe-coffee.png'
logo = cv2.imread(img_path, -1)
# Resize logo img
watermark = utils.img_resize(logo, height = 250)
# Grayscale watermark
# watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
# Change watermark to 4channel
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
water_h, water_w, water_c = watermark.shape
cv2.imshow('watermark', watermark)
'''


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = utils.rescale_frame(frame, percent = 45)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # convert frame with alpha channel
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Find faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Find face roi
        roi_gray = gray[y:y+h, x:x+h]
        roi_colour = frame[y:y+h, x:x+h]
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        # Find eye roi 
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            eroi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eroi_colour = roi_colour[ey:ey+eh, ex:ex+ew]

            # Resize glasses
            rz_glasses = utils.img_resize(glasses.copy(), width=ew)

            # Grab shape of glasses
            gw, gh, gc = rz_glasses.shape

            # Replace face pixels to glasses pixels
            for i in range(0, gw):
                for j in range(0, gh) :
                    if rz_glasses[i, j][3] != 0: # alpha = 0 is transparent
                        frame[i+ey+8, j+ex] = rz_glasses[i, j]

            # cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)

        noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in noses:
            nroi_gray = roi_gray[ny:ny+nh, nx:nx+nh]
            nroi_colour = roi_colour[ny:ny+nh, nx:nx+nw]

            # Resize glasses
            rz_mustache = utils.img_resize(mustache.copy(), width=nw)

            # Grab shape of glasses
            mw, mh, mc = rz_mustache.shape

            # Replace face pixels to glasses pixels
            for i in range(0, mw):
                for j in range(0, mh) :
                    if rz_mustache[i, j][3] != 0: # alpha = 0 is transparent
                        frame[i+ny+25, j+nx] = rz_mustache[i, j]
            # cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (255,0,0), 2)



    # change frame with alpha channel back to bgr
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow('frame',frame)
    # cv2.imshow('region',region)
    # cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()