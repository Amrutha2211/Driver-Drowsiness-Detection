from flask import Flask, render_template, redirect
import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import threading
from pygame import mixer

app = Flask(__name__)

# Initialize pygame mixer
mixer.init()
no_driver_sound = mixer.Sound('static/audio/nodriver_audio.wav')
sleep_sound = mixer.Sound('static/audio/rest_audio.wav')
tired_sound = mixer.Sound('static/audio/sleep_sound.wav')

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    return 'active' if ratio > 0.22 else 'sleep'

def mouth_aspect_ratio(mouth):
    A = compute(mouth[2], mouth[10])
    B = compute(mouth[4], mouth[8])
    C = compute(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

(mStart, mEnd) = (49, 68)

def tired():
    start = time.time()
    rest_time_start = start

    while time.time() - start < 9:
        if time.time() - rest_time_start > 3:
            tired_sound.play()
            time.sleep(1)
            tired_sound.stop()
            rest_time_start = time.time()

def detech():
    sleep_sound_flag = 0
    no_driver_sound_flag = 0
    yawning = 0
    no_yawn = 0
    sleep = 0
    active = 0
    status = ""
    color = (0, 0, 0)
    no_driver = 0
    frame_color = (0, 255, 0)

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    start = time.time()
    no_driver_time = time.time()
    no_driver_sound_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_frame = frame.copy()
        faces = detector(gray, 0)

        if faces:
            no_driver_sound_flag = 0
            no_driver_sound.stop()
            no_driver = 0
            no_driver_time = time.time()

            for face in faces:
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)

                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
                mouth = landmarks[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)

                # Yawning detection
                if mar > 0.70:
                    sleep = 0
                    active = 0
                    yawning += 1
                    status = "Yawning"
                    color = (255, 0, 0)
                    frame_color = (255, 0, 0)
                    sleep_sound_flag = 0
                    sleep_sound.stop()

                # Sleeping detection
                elif left_blink == 'sleep' or right_blink == 'sleep':
                    if yawning > 20:
                        no_yawn += 1
                    sleep += 1
                    yawning = 0
                    active = 0
                    if sleep > 5:
                        status = "Sleeping!"
                        color = (0, 0, 255)
                        frame_color = (0, 0, 255)
                        if sleep_sound_flag == 0:
                            sleep_sound.play()
                        sleep_sound_flag = 1

                else:  # Awake
                    if yawning > 20:
                        no_yawn += 1
                    sleep = 0
                    yawning = 0
                    active += 1
                    status = "Awake"
                    color = (0, 255, 0)
                    frame_color = (0, 255, 0)
                    if active > 5:
                        sleep_sound_flag = 0
                        sleep_sound.stop()

                cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # Tired detection every 60s
                if time.time() - start < 60 and no_yawn >= 3:
                    no_yawn = 0
                    tired()
                elif time.time() - start > 60:
                    start = time.time()

                for n in range(0, 68):
                    x, y = landmarks[n]
                    cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        else:  # No driver detected
            no_driver += 1
            sleep_sound_flag = 0
            sleep_sound.stop()
            if no_driver > 10:
                status = "No Driver"
                color = (0, 0, 0)
            if time.time() - no_driver_time > 5:
                if no_driver_sound_flag == 0:
                    no_driver_sound.play()
                    no_driver_sound_start = time.time()
                else:
                    if time.time() - no_driver_sound_start > 3:
                        no_driver_sound.play()
                        no_driver_sound_start = time.time()
                no_driver_sound_flag = 1

        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.imshow("DRIVER (Enter q to exit)", frame)
        cv2.imshow("68_POINTS", face_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    no_driver_sound.stop()
    sleep_sound.stop()
    tired_sound.stop()
    cap.release()
    cv2.destroyAllWindows()

# Flask routes
@app.route("/open_camera")
def open_camera():
    threading.Thread(target=detech).start()  # Run camera in a separate thread
    return redirect("/")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
