import cv2
import mediapipe as mp
import math as m
import numpy as np
import streamlit as st
from PIL import Image

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    angle = m.degrees(m.atan2(y3 - y2, x3 - x2) - m.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

# Main function to process video
def process_video(file_name):
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    # Read the video
    video_bytes = file_name.read()

    # Play the video
    st.video(video_bytes)

# Streamlit app
def main():
    st.title("Yoga Pose Analysis")

    option = st.radio("Select Source:", ("File Path", "Upload Video", "Webcam"))

    if option == "File Path":
        file_name = st.text_input("Enter the path to the video file:")
        if not file_name:
            st.warning("Please enter the path to the video file.")
            return
        process_video(file_name)

    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video File", type=["mp4"])
        if uploaded_file is not None:
            process_video(uploaded_file)

    elif option == "Webcam":
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            # You can integrate your existing code for pose detection here

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            st.image(rgb_frame, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
