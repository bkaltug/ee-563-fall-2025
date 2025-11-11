import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:

  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result
) -> np.ndarray:
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image


IMAGE_FILE1 = 'images/face_right.png'
IMAGE_FILE2 = 'images/face_left.png'
IMAGE_FILE3 = 'images/face_straight.png'

base_options = python.BaseOptions(model_asset_path='models/face_detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

image1 = mp.Image.create_from_file(IMAGE_FILE1)
image2 = mp.Image.create_from_file(IMAGE_FILE2)
image3 = mp.Image.create_from_file(IMAGE_FILE3)

detection_result1 = detector.detect(image1)
detection_result2 = detector.detect(image2)
detection_result3 = detector.detect(image3)

image_copy1 = np.copy(image1.numpy_view())
image_copy2 = np.copy(image2.numpy_view())
image_copy3 = np.copy(image3.numpy_view())

annotated_image1 = visualize(image_copy1, detection_result1)
annotated_image2 = visualize(image_copy2, detection_result2)
annotated_image3 = visualize(image_copy3, detection_result3)

rgb_annotated_image1 = cv2.cvtColor(annotated_image1, cv2.COLOR_BGR2RGB)
rgb_annotated_image2 = cv2.cvtColor(annotated_image2, cv2.COLOR_BGR2RGB)
rgb_annotated_image3 = cv2.cvtColor(annotated_image3, cv2.COLOR_BGR2RGB)

user_choice = input("Enter 1 for the face looking at right, 2 for left or 3 for straight: ")

image_to_show = None
if user_choice == '1':
    image_to_show = rgb_annotated_image1
    det = detection_result1
elif user_choice == '2':
    image_to_show = rgb_annotated_image2
    det = detection_result2
elif user_choice == '3':
    image_to_show = rgb_annotated_image3
    det = detection_result3
else:
    print("Invalid input. Please enter 1, 2, or 3.")

LEFT_THRESHOLD = 0.1 
RIGHT_THRESHOLD = -0.1  
face_direction = "straight" 

try:
    detection = det.detections[0]
    
    right_eye_x = detection.keypoints[0].x
    left_eye_x = detection.keypoints[1].x
    nose_tip_x = detection.keypoints[2].x
    
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
 
    nose_offset = eye_center_x - nose_tip_x
    
    if nose_offset > LEFT_THRESHOLD:
        face_direction = "left"
    elif nose_offset < RIGHT_THRESHOLD:
        face_direction = "right"
    else:
        face_direction = "straight"

    print(face_direction)

except IndexError:
    print("No face detected.")

if image_to_show is not None:
    cv2.imshow('Detection Result', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()