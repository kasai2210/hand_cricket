from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

model = load_model("model_1.h5")

cap = cv2.VideoCapture(0)  
cap.set(3, 1280)
cap.set(4, 720)

def bowl(n):
    if n==0:
        return 0
    elif n==5:
        return 4
    else:
        return 5

def bat(n):
    return n

start = 0

while True:
    _, frame = cap.read()

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))

    pred = model.predict(np.array([img]))
    ans = int(np.squeeze(np.dot(pred, [0, 1, 2, 3, 4, 5])))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + str(ans),
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + str(bat(ans)),
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    icon = cv2.imread(
        "images/{}.jpg".format(bat(ans)))
    icon = cv2.resize(icon, (400, 400))
    frame[100:500, 800:1200] = icon
            
    cv2.imshow("frame", frame)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()