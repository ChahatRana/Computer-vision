import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

class HandDetector:
    def __init__(self, maxHands=2):
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            h, w, _ = img.shape

            for hand_index, hand in enumerate(result.hand_landmarks):
                for id, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # print ONLY palm (id 0 of first hand)
                    if hand_index == 0 and id == 0:
                        print(id, cx, cy)

                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return img


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = HandDetector(maxHands=2)
    pTime = 0

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.findHands(img, draw=False)

            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime != 0 else 0
            pTime = cTime

            cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Hand Tracking", img)

            # window close OR 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
