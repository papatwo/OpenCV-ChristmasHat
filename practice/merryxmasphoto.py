import numpy as np 
import cv2
import dlib
import utils


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/thirdparty_frontaleyes35x15.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/thirdparty_nose18x15.xml')

glasses = cv2.imread('images/glasses.png', -1)
mustache = cv2.imread('images/mustache.png', -1)
frame = cv2.imread('g.jpg')



# read img with alpha channel by -1 (only for png pic)
hat_im = cv2.imread('hat2.png', -1)

# print video.get(cv2.cv.CV_CAP_PROP_FPS)
# frame = utils.rescale_frame(frame, percent = 50)
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
                    frame[i+ey+10, j+ex] = rz_glasses[i, j]

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
                    frame[i+ny+30, j+nx] = rz_mustache[i, j]
        # cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (255,0,0), 2)



# change frame with alpha channel back to bgr
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

utils.letsHat(frame, hat_im)

# cv2.imshow('photo', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# img show
# cv2.imshow('image', hat_im)
# cv2.imshow('image2', face_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# output = letsHat(face_im, hat_im)
# cv2.imwrite('LetsHat.jpg', output)

