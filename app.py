import streamlit as st
import cv2
import mediapipe as mp
import time
import math as m
import numpy as np
from tempfile import NamedTemporaryFile
import subprocess

# Check if libGL.so.1 is present
def check_libGL():
    try:
        subprocess.check_output(["ldconfig", "-p", "|", "grep", "libGL.so.1"])
        return True
    except subprocess.CalledProcessError:
        return False

# Install libGL.so.1
def install_libGL():
    subprocess.run(["sudo", "apt", "update"])
    subprocess.run(["sudo", "apt", "install", "libgl1-mesa-glx"])

# Check and install libGL.so.1 if needed
if not check_libGL():
    st.warning("libGL.so.1 is missing. Attempting to install it...")
    install_libGL()

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3
    angle = m.degrees(m.atan2(y3-y2, x3-x2) - m.atan2(y1-y2, x1-x2))
    if angle < 0:
        angle += 360
    return angle

# Define Streamlit app
def main():
    st.title("Yoga Pose Analysis")
    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)

        mpDraw = mp.solutions.drawing_utils
        mpPose = mp.solutions.pose
        pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

        good_frames = 0
        bad_frames = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_temp_file = NamedTemporaryFile(delete=False)
        output_temp_path = output_temp_file.name
        video_output = cv2.VideoWriter(output_temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        stframe = st.empty()

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                st.error("Error reading video stream")
                break

            h, w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            lm = keypoints.pose_landmarks

            landmarks = []
            if keypoints.pose_landmarks:
                mpDraw.draw_landmarks(image, keypoints.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for landmark in keypoints.pose_landmarks.landmark:
                    landmarks.append((int(landmark.x * w), int(landmark.y * h)))

            left_knee_angle = calculateAngle(landmarks[mpPose.PoseLandmark.LEFT_HIP.value],
                                             landmarks[mpPose.PoseLandmark.LEFT_KNEE.value],
                                             landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value])
            right_knee_angle = calculateAngle(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value],
                                              landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value])

            angle_text_string = 'left knee : ' + str(int(left_knee_angle)) + '  right knee : ' + str(
                int(right_knee_angle))
            feedback = 'Good Job hold still'
            feedback1 = 'Adjust your knee'

            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                if left_knee_angle > 270 and left_knee_angle < 320 or right_knee_angle > 30 and right_knee_angle < 80:
                    st.write(angle_text_string)
                    st.write(feedback)
                    bad_frames = 0
                    good_frames += 1
                else:
                    st.write(angle_text_string)
                    st.write(feedback1)
                    good_frames = 0
                    bad_frames += 1

            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                st.write(time_string_good)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                st.write(time_string_bad)

            video_output.write(image)
            stframe.image(image, channels="BGR")

        # Close the temporary files after processing
        cap.release()
        video_output.release()
        cv2.destroyAllWindows()
        st.sidebar.text("Uploaded video processed successfully.")

        st.video(output_temp_path)

if __name__ == "__main__":
    main()
