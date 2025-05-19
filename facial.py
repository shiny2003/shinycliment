import cv2
import os
if not os.path.exists('detected_faces'):
    os.makedirs('detected_faces')
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_smile.xml')
saved_count = 0
while cap.isOpened():
    ret, frame =cap . read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for i,(x, y, w, h)in enumerate(faces):
        cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y +h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh),(0, 0, 255), 2)
        face_img = frame[y:y +h, x:x + w]
        cv2.imwrite(f"detected_faces/face_{saved_count}.jpg", face_img)
        saved_count += 1
    cv2.imshow('Face, Eyes, and Smile Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
