import streamlit as st
import cv2
import mediapipe as mp
import time
import math as m
import numpy as np
from tempfile import NamedTemporaryFile

# Calculate distance
def find_distance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle
def find_angle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3
    angle = m.degrees(m.atan2(y3-y2, x3-x2) - m.atan2(y1-y2, x1-x2))
    if angle < 0:
        angle += 360
    return angle

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    good_frames = 0
    bad_frames = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lm = keypoints.pose_landmarks

        landmarks = []
        if keypoints.pose_landmarks:
            mp_draw.draw_landmarks(image, keypoints.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for landmark in keypoints.pose_landmarks.landmark:
                landmarks.append((int(landmark.x * w), int(landmark.y * h)))

        left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        angle_text_string = 'left knee : ' + str(int(left_knee_angle)) + '  right knee : ' + str(
            int(right_knee_angle))
        feedback = 'Good Job hold still'
        feedback1 = 'Adjust your knee'

        if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
            if left_knee_angle > 270 and left_knee_angle < 320 or right_knee_angle > 30 and right_knee_angle < 80:
                good_frames += 1
            else:
                bad_frames += 1

        video_output.write(image)

    cap.release()
    video_output
