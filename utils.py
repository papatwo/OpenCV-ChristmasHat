import cv2
import os
import numpy as np
import dlib

def make_480p():
  cap.set(3, 640)
  cap.set(4, 480)

def rescale_frame(frame, percent = 75):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

def img_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# Get img size
	dim = None
	(h, w) = image.shape[:2]
	# If both w and h are None, return original img
	if width is None and height is None:
		return image
	# Check if width is None, construct dim according to height ratio
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))


	# Resize img
	resized = cv2.resize(image, dim, interpolation = inter)

	# Return
	return resized



def accessory(frame, roi):
	face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascades/thirdparty_frontaleyes35x15.xml')
	nose_cascade = cv2.CascadeClassifier('haarcascades/thirdparty_nose25x15.xml')

	glasses = cv2.imread('images/glasses.png', -1)
	mustache = cv2.imread('images/mustache.png', -1)





	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # convert frame with alpha channel
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Find faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Find eye roi 
	eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

	for (ex, ey, ew, eh) in eyes:
	    eroi_gray = roi[ey:ey+eh, ex:ex+ew]

	    # Resize glasses
	    rz_glasses = img_resize(glasses.copy(), width=ew)

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
	    nroi_gray = roi[ny:ny+nh, nx:nx+nh]

	    # Resize glasses
	    rz_mustache = img_resize(mustache.copy(), width=nw)

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
	return frame






def letsHat(img, hat):

	r, g, b, a = cv2.split(hat)
	hat_rgb = cv2.merge((r, g, b))

	# Trained dlib face key points detector
	predictor_path = "shape_predictor_5_face_landmarks.dat"
	predictor = dlib.shape_predictor(predictor_path)

	# Dlib front face detector
	detector = dlib.get_frontal_face_detector()

	# Front face detection result
	face_detect = detector(img, 1)

	# If face detected
	if len(face_detect) > 0:
		for d in face_detect: # for each face detected
			x, y, w, h = d.left(), d.top(), (d.right() - d.left()), (d.bottom() - d.top())
			roi = img[x:x+w, y:y+h]
			# cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),3)
			# img = accessory(img, roi)
			# imgRect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)

			# 5 key points detection 
			shape = predictor(img, d)
			# for point in shape.parts():
			# 	face_pts = cv2.circle(img, (point.x, point.y), 3, color = (0, 255, 0))
			# 	Draw 5 feature pts one by one
			# 	cv2.imshow('image', face_pts)
			# 	cv2.waitKey(0)
			# 	cv2.destroyAllWindows()

			## Draw detection retangle and pts on face
			# cv2.imshow('image', img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# Select outermost feature pts on left and right eyes
			pt1 = shape.part(0)
			pt2 = shape.part(2)
			# print pt1, pt2

			# Calculate centre pt of eye
			centre_pt = ((pt1.x + pt2.x) // 2, (pt1.y + pt2.y) // 2)
			# face_centrept = cv2.circle(img, centre_pt, 3, color = (0, 255, 0))

			## Draw centre pts on face
			# cv2.imshow('image', img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# Adjust hat size according to face size
			# # shape[0]=width, shape[1]=height
			# print img.shape[0], img.shape[1]
			factor = 1.5
			resize_hat_h = int(round(hat_rgb.shape[0] * w / hat_rgb.shape[1] * factor))
			resize_hat_w = int(round(hat_rgb.shape[1] * w / hat_rgb.shape[1] * factor))

			if resize_hat_h > y:
				resize_hat_h = y -1

			resize_hat = cv2.resize(hat_rgb, (resize_hat_w, resize_hat_h))
			# cv2.imshow('image', resize_hat)
			# cv2.imshow('image2', hat_rgb)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# Make resize mask from alpha channel
			mask = cv2.resize(a, (resize_hat_w, resize_hat_h))
			mask_inv = cv2.bitwise_not(mask)

			# Hat skew wrt face detection rectangle
			dh = 0
			dw = 0

			# ROI of figure image
			roi = img[(y + dh - resize_hat_h) : (y + dh), (x + dw) : (x + resize_hat_w + dw)]
			# imgRect = cv2.rectangle(img, (x + dw, y + dh - resize_hat_h), (x + resize_hat_w + dw, y + dh), (255, 0, 0), 2, 8, 0)
			# imgRect = cv2.rectangle(img, (x + dw, y + dh), (x + dw, y + dh), (0, 2, 0), 2, 8, 0)
			# cv2.imshow('image', img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			roi = img[(y + dh - resize_hat_h) : (y + dh), \
						(centre_pt[0] - resize_hat_w // 3) : (centre_pt[0] + resize_hat_w // 3 * 2)]

			# Extract hat space in ROI
			roi = roi.astype(float)
			# print mask_inv
			mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
			alpha = mask_inv.astype(float) / 255
			# print alpha
			if alpha.shape != roi.shape:
				alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))

			bg = cv2.multiply(alpha, roi)
			bg = bg.astype('uint8')

			# cv2.imshow('imge', bg)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# Extract hat region
			hat_region = cv2.bitwise_and(resize_hat, resize_hat, mask = mask)
			cv2.imwrite('hat.jpg', hat_region)
			# cv2.imshow('image', hat_region)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# print bg.shape, hat_region.shape
			if bg.shape != hat_region.shape:
				hat_region = cv2.resize(hat_region, (bg.shape[1], bg.shape[0]))

			# Add the two ROI (add hat to background image)
			add_hat = cv2.add(bg, hat_region)
			# cv2.imshow('addhat',add_hat)
			# cv2.imshow('hat', hat_region)
			# cv2.imshow('bg', bg)
			# cv2.imshow('original', img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			# Put the hat added region back to original image
			img[(y + dh - resize_hat_h) : (y + dh), \
					(centre_pt[0] - resize_hat_w // 3) : (centre_pt[0] + resize_hat_w // 3 * 2)]\
			= add_hat

		# Show the result and save
		cv2.imshow('original', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows() 
		# return img

	else:
		print "No Face Detected!!!"
		cv2.imshow('original', img)
