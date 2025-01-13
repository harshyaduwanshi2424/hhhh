import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon= True, trackCon= True):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=False):
        lmlist = []
        results = self.pose.process(img)  # Corrected variable name
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist

    def findAngle(self, img, p1, p2, p3, draw=False):
        lmlist = self.findPosition(img, False)  # Get landmarks
        x1, y1 = lmlist[p1][1:]  # Corrected to use lmlist
        x2, y2 = lmlist[p2][1:]
        x3, y3 = lmlist[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
        return angle


def main():
        cap = cv2.VideoCapture("curl.mp4")  # Change "test.mp4" to the path of your video file
        detector = poseDetector()
        count = 0
        dir = 0
        pTime = 0

        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.resize(img, (1288, 720))

            img = detector.findPose(img, False)
            lmlist = detector.findPosition(img, False)

            if len(lmlist) != 0:
                # right arm
                angle = detector.findAngle(img, 12, 14, 16)
                # left arm
                angle = detector.findAngle(img, 11, 13, 15)

                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (650, 100))  # min and max value of the bar (OpenCV convention)

                color = (255, 0, 255)
                if per == 100:
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                print(count)

                # Next 3 lines for the bar
                cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

                # For curl count
                cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.imshow("Image", img)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
        main()
