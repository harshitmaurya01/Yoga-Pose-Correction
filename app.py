import cv2
import mediapipe as mp
import time
import math as m
import numpy as np
import streamlit as st

# Constants
GOOD_POSTURE_THRESHOLD = 165
BAD_POSTURE_THRESHOLD = 195
KNEE_ANGLE_THRESHOLD = 30
HIP_ANGLE_THRESHOLD = 80
BAD_POSTURE_WARNING_TIME = 60

# Function to calculate distance
def find_distance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate angle
def find_angle(x1, y1, x2, y2):
    return m.degrees(m.atan2(y2 - y1, x2 - x1))

# Function to draw text on image
def draw_text(image, text, x, y, color, font_scale):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

# Function to process image
def process_image(image):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with MediaPipe Pose
    keypoints = pose.process(image)

    # Convert image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return keypoints

# Function to calculate knee angles
def calculate_knee_angles(landmarks):
    left_knee_angle = find_angle(landmarks[mpPose.PoseLandmark.LEFT_HIP.value][0], landmarks[mpPose.PoseLandmark.LEFT_HIP.value][1],
                                 landmarks[mpPose.PoseLandmark.LEFT_KNEE.value][0], landmarks[mpPose.PoseLandmark.LEFT_KNEE.value][1],
                                 landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value][0], landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value][1])

    right_knee_angle = find_angle(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value][0], landmarks[mpPose.PoseLandmark.RIGHT_HIP.value][1],
                                 landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value][0], landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value][1],
                                 landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value][0], landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value][1])

    return left_knee_angle, right_knee_angle

# Function to determine posture
def determine_posture(knee_angles):
    if knee_angles[0] > GOOD_POSTURE_THRESHOLD and knee_angles[0] < BAD_POSTURE_THRESHOLD or \
       knee_angles[1] > GOOD_POSTURE_THRESHOLD and knee_angles[1] < BAD_POSTURE_THRESHOLD:
        return "Good"
    else:
        return "Bad"

# Function to send warning
def send_warning():
    st.error("Bad posture detected! Please adjust your position.")

# Streamlit app
st.title("Vrikshasna Posture Detection")

# Load video file
video_file = st.file_uploader("Select a video file", type=["mp4"])

if video_file:
    # Initialize video capture
    cap = cv2.VideoCapture(video_file)

    # Initialize MediaPipe Pose
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Initialize frame counters
    good_frames = 0
    bad_frames = 0

    while cap.isOpened():
        # Capture frame
        success, image = cap.read()
        if not success:
            break

        # Process image
        keypoints = process_image(image)

        # Calculate knee angles
        landmarks = []
        if keypoints.pose_landmarks:
            for landmark in keypoints.pose_landmarks.landmark:
                landmarks.append((int(landmark.x * image
