import os
import numpy as np
import cv2
from PIL import Image
import pickle



base_dir = os.path.dirname(os.path.abspath(__file__)) # wherever this py file is saved, os.path gives the current location
image_dir = os.path.join(base_dir, "images")

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


y_labels = []
x_train = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            # print path
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            # print label, path

            # Create label-id dictionary
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            # print label_ids

            # Load image
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")

            # Region of interest in training data
            face = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in face:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print np.array(y_labels)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

# Code for training recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner_resize.yml")




