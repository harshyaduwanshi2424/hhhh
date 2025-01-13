# Importing modules for Tkinter
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Importing modules for mediapipe
import cv2
from posedetector import PoseDetector
import numpy as np
import time


# Accessing Video


def clear():
    global counter
    counter = 0


def camselect(cam=1, file1=NONE):
    global cap

    if cam:

        cap = cv2.VideoCapture(0)
        clear()

    else:
        cap = cv2.VideoCapture(file1)
        if cap == None:
            camvideo()

        clear()


camselect()

detector = PoseDetector()

counter = 0
stage = None
pTime = 0


def test2():
    global counter
    global pTime
    global stage
    global t

    success, img = cap.read()
    img = cv2.resize(img, (990, 640))

    img = detector.findPose(img)

    hipL = detector.hipL()
    kneeL = detector.kneeL()
    ankleL = detector.ankleL()
    hipR = detector.hipR()
    kneeR = detector.kneeR()
    ankleR = detector.ankleR()

    angle_knee_L = 180
    angle_knee_R = 180

    if (hipL and kneeL and ankleL):

        angle_knee_L = round(detector.calculate_angle(hipL, kneeL, ankleL), 2)
        angle_knee_R = round(detector.calculate_angle(hipR, kneeR, ankleR), 2)

        if angle_knee_L > 150 and angle_knee_R > 150:
            stage = "UP"
        if angle_knee_L <= 90 and stage == 'UP' and angle_knee_R <= 90:
            stage = "DOWN"
            counter += 1

    cv2.putText(img, f'{angle_knee_L}*', (600, 300), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f'{angle_knee_R}*', (200, 300), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 2, cv2.LINE_AA)

    # Rep data
    cv2.putText(img, 'REPS', (50, 500),
                cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(counter),
                (50, 550),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(img, 'STAGE', (150, 500),
                cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, stage,
                (150, 550),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Displaying FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb


# Tkinter Code


def openfile():
    file1 = filedialog.askopenfilename()
    cam = 0
    camselect(cam, file1)


def openV1():
    file1 = '2.mp4'
    cam = 0
    camselect(cam, file1)


def camvideo():
    cam = 1
    camselect(cam)


def close():
    window.destroy()

def run_posee():
    window.destroy()
    import cv2
    import numpy as np
    import math
    from cvzone.PoseModule import PoseDetector
    from filterpy.kalman import KalmanFilter

    cap = cv2.VideoCapture(0)
    pd = PoseDetector(detectionCon=0.70, trackCon=0.70)

    class AngleFinder:
        def __init__(self, lmlist, p1, p2, p3, p4, p5, p6, drawpoints):
            self.lmlist = lmlist
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
            self.p4 = p4
            self.p5 = p5
            self.p6 = p6
            self.drawpoints = drawpoints

        def angle(self):
            if self.lmlist:
                point1 = self.lmlist[self.p1][1:]
                point2 = self.lmlist[self.p2][1:]
                point3 = self.lmlist[self.p3][1:]
                point4 = self.lmlist[self.p4][1:]
                point5 = self.lmlist[self.p5][1:]
                point6 = self.lmlist[self.p6][1:]

                x1, y1 = point1
                x2, y2 = point2
                x3, y3 = point3
                x4, y4 = point4
                x5, y5 = point5
                x6, y6 = point6

                leftAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                rightAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

                leftAngle = int(np.interp(leftAngle, [-170, 180], [100, 0]))
                rightAngle = int(np.interp(rightAngle, [-50, 20], [100, 0]))

                if self.drawpoints:
                    cv2.circle(img, (x1, y1), 10, (0, 255, 255), 5)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 4)
                    cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 4)
                    cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 4)
                    cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 4)

                return leftAngle, rightAngle

    # Initialize Kalman filters
    kf_left = KalmanFilter(dim_x=1, dim_z=1)
    kf_right = KalmanFilter(dim_x=1, dim_z=1)

    kf_left.x = np.array([0.0])
    kf_right.x = np.array([0.0])

    kf_left.F = np.array([[1]])
    kf_right.F = np.array([[1]])

    kf_left.H = np.array([[1]])
    kf_right.H = np.array([[1]])

    kf_left.P *= 1e2
    kf_right.P *= 1e2

    # for counting and direction
    score = 0
    dir = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        img = cv2.resize(img, (640, 480))
        pd.findPose(img, draw=0)
        lmlist, bbox = pd.findPosition(img, draw=0)

        angles = AngleFinder(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=False)
        left, right = angles.angle()

        # Apply Kalman filter
        kf_left.predict()
        kf_left.update(left)
        left_filtered = int(kf_left.x[0])

        kf_right.predict()
        kf_right.update(right)
        right_filtered = int(kf_right.x[0])

        print(left_filtered, right_filtered)

        if left_filtered >= 90 and right_filtered >= 90:
            if dir == 0:
                score += 0.25
                dir = 1
        if left_filtered <= 90 and right_filtered <= 90:
            if dir == 1:
                score += 0.25
                dir = 0


        rightval = np.interp(right, [0, 100], [400, 200])               #maybe try 0,100
        leftval = np.interp(left, [0, 100], [400, 200])


        cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
        cv2.putText(img, str(int(score)), (1, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 6)

        cv2.putText(img, 'R', (24, 194), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
        cv2.rectangle(img, (8, 200), (50, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 255, 0), -1)

        cv2.putText(img, 'L', (604, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
        cv2.rectangle(img, (582, 200), (632, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (582, int(leftval)), (632, 400), (255, 255, 0), -1)

        # bars
        cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1100, int(rightval)), (1175, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(right)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)

        valueleft = np.interp(left, [0, 100], [0, 100])
        valueright = np.interp(right, [0, 100], [400, 200])

        if valueleft >= 70:
            cv2.rectangle(img, (582, int(leftval)), (632, 400), (0, 0, 255), -1)

        if valueright >= 70:
            cv2.rectangle(img, (582, int(rightval)), (632, 400), (0, 0, 255), -1)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) == 27:  # Break the loop on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

    # Function to run main.py


def run_main():
    window.destroy()

    import cv2
    import numpy as np
    import math
    import time
    from posemodule import poseDetector  # Import your pose module class
    from filterpy.kalman import KalmanFilter

    def start_processing(choice):
        cap = None

        if choice == '1':
            cap = cv2.VideoCapture(0)  # Use webcam (0) as the video source
        elif choice == '2':
            cap = cv2.VideoCapture('curls2.mp4')  # Use curls2.mp4 as the video source
        else:
            print("Invalid choice. Exiting...")
            exit()

        detector = poseDetector()
        count = 0
        dir = 0
        pTime = 0

        # Initialize Kalman filter for the angle
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([0.0])
        kf.F = np.array([[1]])
        kf.H = np.array([[1]])
        kf.P *= 1e2

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture frame. Exiting...")
                break

            img = cv2.resize(img, (1288, 720))
            img = detector.findPose(img, False)
            lmlist = detector.findPosition(img, False)

            if len(lmlist) != 0:
                # right arm
                angle = detector.findAngle(img, 12, 14, 16)  # knowledge base

                # Apply Kalman filter
                kf.predict()
                kf.update(angle)
                angle_filtered = int(kf.x[0])

                per = np.interp(angle_filtered, (210, 310), (0, 100))
                bar = np.interp(angle_filtered, (220, 310),
                                (650, 100))  # min and max value of the bar (OpenCV convention)

                color = (255, 0, 255)
                if per == 100:  # rules
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
            if cv2.waitKey(1) == 27:  # Break the loop on ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_user_choice():
        root = Tk()
        root.title("Choose Video Source")

        def on_button_click(choice):
            root.destroy()
            start_processing(choice)

        label = Label(root, text="Choose video source:")
        label.pack(pady=10)

        button_webcam = Button(root, text="Webcam", command=lambda: on_button_click('1'))
        button_webcam.pack(pady=5)

        button_video = Button(root, text="Video File (curls2.mp4)", command=lambda: on_button_click('2'))
        button_video.pack(pady=5)

        root.mainloop()

    if __name__ == "__main__":
        get_user_choice()


# Window Creation
window = Tk()
window.configure(bg='blue')
window.title("Pose Detector")
width = window.winfo_screenwidth() + 10
height = window.winfo_screenheight() + 10
window.geometry("%dx%d" % (width, height))
window.minsize(width, height)
window.maxsize(width, height)

## Design
mainlabel = Label(window, text="Exercise Counter", font=(
    "Raleway", 20, "bold", "italic"), bg="blue", fg='yellow')
mainlabel.pack()

f1 = Frame(window, bg='blue')
f1.pack(side=LEFT, fill='y', anchor='nw')



explore = Button(f1, text="Browse File", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=openfile).pack(padx=50, pady=10)

livecam = Button(f1, text="Open Web Cam", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=camvideo).pack(pady=10)

v1 = Button(f1, text="Test Video 1", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=openV1).pack(padx=50, pady=10)

CLR = Button(f1, text="CLear Screen", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=clear).pack(padx=50, pady=10)

pose = Button(f1, text="Shoulder Press", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=run_posee).pack(padx=50, pady=10)
curls = Button(f1, text="Biceps", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=run_main).pack(padx=50, pady=10)

Exit_Application = Button(f1, text="Exit the Application", bg='blue', fg='yellow', font=(
    "Calibri", 14, "bold"), command=close).pack(pady=50)





#Video


label1 = Label(window, width=960, height=640)
label1.place(x=240, y=50)


def select_img():
    image = Image.fromarray(test2())
    finalImage = ImageTk.PhotoImage(image)
    label1.configure(image=finalImage)
    label1.image = finalImage
    window.after(1, select_img)


select_img()
window.mainloop()