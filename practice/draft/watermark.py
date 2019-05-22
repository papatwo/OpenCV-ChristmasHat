import numpy as np
import cv2
import utils


cap = cv2.VideoCapture(0)
# width = cap.get(3)
# height = cap.get(4)

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



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = utils.rescale_frame(frame, percent = 50)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cv2.rectangle(frame,(50,150),(75,175),(255,0,0),2)
    # Detect faces by using the cascades
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find a certain area in the frame
    start_x = 0
    start_y = 0
    w = water_w
    h = water_h
    end_x = start_x + w
    end_y = start_y + h
    colour = (0, 255, 0)
    stroke = 2
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colour, stroke)

    # Print out pixels values in that area
    region = frame[start_x:end_x, start_y:end_y]

    # Overlay with 4 channels BGR and Alpha
    frame_h, frame_w, frame_c = frame.shape
    overlay = np.zeros((frame_h, frame_w, 4), dtype = 'uint8')
    # cv2.imshow('overlay', overlay)

    for i in range(0, water_h):
        for j in range(0, water_w):
            if watermark[i, j][3] != 0:
                overlay[frame_h-water_h-10+i, frame_w-water_w-10+j] = watermark[i, j]

    # Why do we create overlay layer: can adjust transparency percentage of added stuff
    # rather than change the raw pixels in the frame
    cv2.addWeighted(overlay, 0.35, frame, 1.0, 0, frame)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    # frame[start_y:end_y, start_x:end_x] = watermark[:,:]
    cv2.imshow('frame',frame)
    # cv2.imshow('region',region)
    # cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()