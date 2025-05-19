# # Virtual Try-On for E-Commerce ...sunglass input

import cv2
import numpy as np

# Overlay function
def overlay_image(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))

    if fg.shape[2] == 4:  # Transparent image
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                fg[:, :, c] * alpha + bg[y:y+h, x:x+w, c] * (1.0 - alpha)
            )
    else:  # No transparency
        fg_rgb = fg[:, :, :3]
        bg[y:y+h, x:x+w] = fg_rgb

    return bg

# Load accessory image
accessory = cv2.imread("glass1.jpeg", cv2.IMREAD_UNCHANGED)
if accessory is None:
    print("Accessory image not found!")
    exit()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        y_offset = y + int(h / 4.5)  # Adjust y position for better fitting
        frame = overlay_image(frame, accessory, x, y_offset, w, int(h / 3.5))

    cv2.imshow("Virtual Try-On", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()




# Virtual Try-On for E-Commerce ...Earrings input

import cv2
import numpy as np

def overlay_image(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h))

    if fg.shape[2] == 4:  # Transparent background
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                fg[:, :, c] * alpha + bg[y:y+h, x:x+w, c] * (1.0 - alpha)
            )
    else:
        bg[y:y+h, x:x+w] = fg[:, :, :3]

    return bg

# Load earrings (both earrings in one image)
earrings_img = cv2.imread("earing.jpeg", cv2.IMREAD_UNCHANGED)
if earrings_img is None:
    print("Earrings image not found!")
    exit()

# Split earrings into left and right halves
h, w = earrings_img.shape[:2]
left_earring = earrings_img[:, :w//2]
right_earring = earrings_img[:, w//2:]

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Approximate ear positions
        left_x = x - int(w * 0.2)
        right_x = x + w
        ear_y = y + int(h * 0.5)
        ear_w = int(w * 0.2)
        ear_h = int(h * 0.4)

        if left_x >= 0 and ear_y + ear_h <= frame.shape[0]:
            frame = overlay_image(frame, left_earring, left_x, ear_y, ear_w, ear_h)
        if right_x + ear_w <= frame.shape[1] and ear_y + ear_h <= frame.shape[0]:
            frame = overlay_image(frame, right_earring, right_x, ear_y, ear_w, ear_h)

    cv2.imshow("Virtual Earrings Try-On", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
