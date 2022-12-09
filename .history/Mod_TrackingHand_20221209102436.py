import cv2 # to read image
import mediapipe as mp # to detect hand
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode # False for static image, True for video
        self.maxHands = maxHands #self -> object
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            # to check if we have multiple hand
            for handLms in self.results.multi_hand_landmarks:
                if draw: # to draw lines
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS) 
        return img 

    # to find position of landmarks
    def findPosition(self, img, handNo=0, draw=True, color=(255, 0, 255), z_axis=False):

        lmList = [] # list of landmarks
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
             #   print(id, lm)
                h, w, c = img.shape # height, width, channels
                if z_axis == False:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                elif z_axis:
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z, 3)
                    # print(id, cx, cy, cz)
                    lmList.append([id, cx, cy, cz]) 

                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    # object of class
    detector = handDetector(maxHands=1) # to detect only one hand
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, z_axis=True, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
