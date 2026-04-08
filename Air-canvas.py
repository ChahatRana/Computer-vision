import numpy as np
import cv2
from collections import deque
from HandTrackingModule import Tracker   # 👈 your hand tracking module

# ================= TRACKER =================
tracker = Tracker()

# ================= DRAW DATA =================
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# ================= CANVAS =================
paintWindow = np.zeros((471,636,3)) + 255

cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2)

cap = cv2.VideoCapture(0)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # 👇 HAND TRACKING
    frame = tracker.hand_landmark(frame)
    frame, dist, x, y = tracker.tracking(frame)

    center = None

    # 👇 REPLACING COLOR CENTER WITH FINGER
    if x != -1 and y != -1:
        center = (x, y)

    # 👇 ONLY DRAW WHEN PINCH (VERY IMPORTANT)
    if center is not None and dist < 40:

        if center[1] <= 65:

            if 40 <= center[0] <= 140:  # CLEAR
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255

            elif 160 <= center[0] <= 255:
                colorIndex = 0

            elif 275 <= center[0] <= 370:
                colorIndex = 1

            elif 390 <= center[0] <= 485:
                colorIndex = 2

            elif 505 <= center[0] <= 600:
                colorIndex = 3

        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    else:
        # 👇 NEW STROKE WHEN HAND LIFTED
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # ================= DRAW =================
    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Tracking (Hand)", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()