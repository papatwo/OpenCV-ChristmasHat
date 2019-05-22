import cv2
import os
import numpy as np

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
		width = int(w * r)
		dim = (width, height)
	else:
		r = width / float(w)
		height = int(h * r)
		dim = (width, height)


	# Resize img
	resized = cv2.resize(image, dim, interpolation = inter)

	# Return
	return resized