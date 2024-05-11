import cv2
import mediapipe as mp
import math
import streamlit as st
from io import BytesIO
import tempfile

# Calculate angle
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

# Initialize mediapipe pose class
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Streamlit app
st.title('Yoga Pose Analysis')

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)
    
    landmarks = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        keypoints = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if keypoints.pose_landmarks:
            mpDraw.draw_landmarks(image, keypoints.pose_landmarks, mpPose.POSE_CONNECTIONS)

            landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in
                         keypoints.pose_landmarks.landmark]

        if len(landmarks) > 0:
            # Calculate angles
            left_knee_angle = calculateAngle(landmarks[mpPose.PoseLandmark.LEFT_HIP.value],
                                              landmarks[mpPose.PoseLandmark.LEFT_KNEE.value],
                                              landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value])

            right_knee_angle = calculateAngle(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value],
                                               landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value],
                                               landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value])

            # Display angles
            st.write('Left Knee Angle:', round(left_knee_angle, 2))
            st.write('Right Knee Angle:', round(right_knee_angle, 2))
            
    cap.release()
