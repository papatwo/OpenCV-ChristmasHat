import numpy as np 
import cv2
import dlib

def letsHat(img, hat):

	# Get alpha channel for the hat img
	r, g, b, a = cv2.split(hat_im)
	hat_rgb = cv2.merge((r, g, b))

	cv2.imwrite("hat_alpha.jpg", a)
	cv2.imwrite("hat_rgb.jpg", hat_rgb)

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
			# imgRect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)

			# 5 key points detection 
			shape = predictor(img, d)
			# for point in shape.parts():
				# face_pts = cv2.circle(img, (point.x, point.y), 3, color = (0, 255, 0))
				# Draw 5 feature pts one by one
				# cv2.imshow('image', face_pts)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()

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
		cv2.waitKey(0)
		cv2.destroyAllWindows() 
		return img

	else:
		print "No Face Detected!!!"





# read img with alpha channel by -1 (only for png pic)
hat_im = cv2.imread('hat2.png', -1)

# read test img with front face
face_im = cv2.imread('test4.jpg')

# img show
# cv2.imshow('image', hat_im)
# cv2.imshow('image2', face_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

output = letsHat(face_im, hat_im)
cv2.imwrite('LetsHat.jpg', output)

