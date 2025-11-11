import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

user_choice = input("Please enter the image path:  ")

base_options = python.BaseOptions(model_asset_path='models/pose_detector.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
image = mp.Image.create_from_file(user_choice)
detection_result = detector.detect(image)

pose_landmarks_list = detection_result.pose_landmarks
arm_status = "None"
VISIBILITY_THRESHOLD = 0.5 

if pose_landmarks_list:
    landmarks = pose_landmarks_list[0]

    left_arm_up = False
    right_arm_up = False

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]


    if left_wrist.visibility > VISIBILITY_THRESHOLD and left_shoulder.visibility > VISIBILITY_THRESHOLD:

        if left_wrist.y < left_shoulder.y:
            left_arm_up = True
            

    if right_wrist.visibility > VISIBILITY_THRESHOLD and right_shoulder.visibility > VISIBILITY_THRESHOLD:
  
        if right_wrist.y < right_shoulder.y:
            right_arm_up = True

    if left_arm_up and right_arm_up:
        arm_status = "both"
    elif left_arm_up:
        arm_status = "left"
    elif right_arm_up:
        arm_status = "right"
    else:
        arm_status = "None"

print(arm_status)

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('Pose Result',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()