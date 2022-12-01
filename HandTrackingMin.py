import cv2
import mediapipe as mp
import time

# to check frame rate
time.sleep(20)
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # parameters already difined
mpDrawLine = mp.solutions.drawing_utils  # to draw multiple lines
prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # processing rgb IMG
    # print(results.multi_hand_landmarks)

    # to check if we have multiple hand
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                # print(id,lm)
                # c-> channels of img
                height, width, c = img.shape
                # position of centre
                cx, cy = int(lm.x * width), int(lm.y * height)
                print(id, cx, cy)

                if id == 9:  # circle for id 1
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDrawLine.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)  # for single hand
            # mpHands.HAND_CONNECTIONS: for connection with dots

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
