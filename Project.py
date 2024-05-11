

import cv2
import mediapipe as mp
import time
import math as m
import numpy as np



# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


def calculateAngle(landmark1, landmark2, landmark3):
        
    x1,y1, =landmark1
    x2,y2, =landmark2
    x3,y3, =landmark3


    angle= m.degrees(m.atan2(y3-y2,x3-x2)-m.atan2(y1-y2,x1-x2))


    if angle < 0 :
        
        angle+= 360

    return angle




landmark_names = {
    0: 'Nose',
    1: 'Left Eye Inside',
    2: 'Left Eye',
    3: 'Left Eye Outside',
    4: 'Right Eye Inside',
    5: 'Right Eye',
    6: 'Right Eye Outside',
    7: 'Left Ear',
    8: 'Right Ear',
    9: 'Mouth Left',
    10: 'Mouth Right',
    11: 'Left Shoulder',
    12: 'Right Shoulder',
    13: 'Left Elbow',
    14: 'Right Elbow',
    15: 'Left Wrist',
    16: 'Right Wrist',
    17: 'Left Pinky Finger',
    18: 'Right Pinky Finger',
    19: 'Left Index Finger',
    20: 'Right Index Finger',
    21: 'Left Thumb',
    22: 'Right Thumb',
    23: 'Left Hip',
    24: 'Right Hip',
    25: 'Left Knee',
    26: 'Right Knee',
    27: 'Left Ankle',
    28: 'Right Ankle',
    29: 'Left Heel',
    30: 'Right Heel',
    31: 'Left Foot Index',
    32: 'Right Foot Index'
}


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# ===============================================================================================#


if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = "D:\Harshit\Project\YogaVidCollected\Dataset\Santosh_vriksh.mp4"
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)


    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mpPose.PoseLandmark


        landmarks=[]
        if keypoints.pose_landmarks:
            mpDraw.draw_landmarks(image,keypoints.pose_landmarks,mpPose.POSE_CONNECTIONS)
            '''
            for id,lm, in enumerate(keypoints.pose_landmarks.landmark):
                print(id,landmark_names[id])
                print(lm)
            '''
            for landmark in keypoints.pose_landmarks.landmark:
                landmarks.append((int(landmark.x*w),int(landmark.y*h)))


        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.
        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)      
        # Left knee.
        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
        # Right knee
        r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        # Right hip.
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
        # Left ankle.
        l_ank_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        l_ank_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
        # Right ankle.
        r_ank_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        r_ank_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)


        # Calculate distance between left shoulder and right shoulder points.
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        '''
        if offset > 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)'''
        
        # Calculate angles.
        left_knee_angle= calculateAngle(landmarks[mpPose.PoseLandmark.LEFT_HIP.value],landmarks[mpPose.PoseLandmark.LEFT_KNEE.value],landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value])

        right_knee_angle= calculateAngle(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value],landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value],landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value])
        '''
        left_knee_angle = findAngle(l_ank_x, l_ank_y, l_hip_x, l_hip_y)
        right_knee_angle = findAngle(r_ank_x, r_ank_y, r_hip_x, r_hip_y)

        print(left_knee_angle)
        print(right_knee_angle)
        '''
        angle_text_string = 'left knee : ' + str(int(left_knee_angle)) + '  right knee : ' + str(int(right_knee_angle))
        feedback = 'Good Job hold still'
        feedback1 = 'Adjust your knee'


         # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

            if left_knee_angle > 270 and left_knee_angle < 320 or right_knee_angle > 30 and right_knee_angle < 80:

                bad_frames = 0
                good_frames += 1
                
                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(left_knee_angle)), (l_knee_x + 10, l_knee_y), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(right_knee_angle)), (r_knee_x + 10, r_knee_y), font, 0.9, light_green, 2)
                cv2.putText(image, feedback, (10, 90), font, 0.9, green, 2)



            else:
                good_frames = 0
                bad_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                cv2.putText(image, str(int(left_knee_angle)), (l_knee_x + 10, l_knee_y), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(right_knee_angle)), (r_knee_x + 10, r_knee_y), font, 0.9, light_green, 2)
                cv2.putText(image, feedback1, (10, 90), font, 0.9, red, 2)

               

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, 60), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, 60), font, 0.9, red, 2)

        # If you stay in bad posture for more than 1 minute (60s) send an alert.
        if bad_time > 60:
            sendWarning()
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('Vrikshasna', image)
        cv2.resizeWindow('Vrikshasna', width, height)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()