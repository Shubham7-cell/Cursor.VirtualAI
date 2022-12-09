import cv2
import mediapipe as mp
import time

# to check frame rate
# time.sleep(20)


class handDetector():
    def __int__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)  # parameters already difined
        self.mpDrawLine = mp.solutions.drawing_utils  # to draw multiple lines

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)  # processing rgb IMG
        # print(results.multi_hand_landmarks)

        # to check if we have multiple hand
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                if draw:
                    self.mpDrawLine.draw_landmarks(
                        img, handlms, self.mpHands.HAND_CONNECTIONS)  # for single hand
                    # mpHands.HAND_CONNECTIONS: for connection with dots
        return img
        # for id, lm in enumerate(handlms.landmark):
        #     # print(id,lm)
        #     # c-> channels of img
        #     height, width, c = img.shape
        #     # position of centre
        #     cx, cy = int(lm.x * width), int(lm.y * height)
        #     print(id, cx, cy)
        #
        #     if id == 4:  # circle for id 1
        #         cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()  # obj
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        # frame rate
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        # display on screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        # 3->thickness, (255,0,255)->color,
        # to capture img
        cv2.imshow("Image", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
