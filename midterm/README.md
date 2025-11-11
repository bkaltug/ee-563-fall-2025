This file contains information about face and pose detector files.

-- FACE DETECTOR--

# Functions

_normalized_to_pixel_coordinates: MediaPipe provides coordinates in a normalized format. This function converts those normalized values into actual pixel coordinates based on the image's width and height to be able to draw in a pop-up.

visualize: This function is responsible for drawing the results on the image. It takes the original image and the detection_result, then uses OpenCV to draw the green keypoint circles and the red bounding box onto a copy of the image.

# Algorithm

This script detects a face in an image and classifies its direction as "left", "right", or "straight".

The main algorithm works by analyzing the keypoints of the detected face, specifically the horizontal positions of the eyes and nose.

1.  The script first loads the pre-trained 'face_detector.tflite' model from MediaPipe.
2.  It runs the detector on the input image, which returns a 'detection_result'. This result contains the bounding box for the face and a list of 6 keypoints.
3. We get the normalized x coordinates for three specific keypoints:
    keypoints[0] = Right Eye
    keypoints[1] = Left Eye
    keypoints[2] = Nose Tip
4. The algorithm calculates the horizontal center of the eyes. It then finds the nose_offset by seeing how far the nose_tip_x is from this center.
5.  The direction is classified by comparing this offset to two asymmetric thresholds:
     If the nose_offset is greater than the LEFT_THRESHOLD, it means the nose is far to the left of the eye center, and the script outputs left.
     If the nose_offset is less than a negative RIGHT_THRESHOLD, it means the nose is far to the right, and the script outputs right.
     If the offset falls between these two thresholds, the nose is considered centered, and the script outputs straight.

-- POSE DETECTOR--

# Function

draw_landmarks_on_image:  This function is responsible for visualizing the pose. It takes the original image and the 'detection_result' and uses MediaPipe's built-in 'solutions.drawing_utils.draw_landmarks' function to draw all 33 landmarks and the connecting lines onto the image.

# Algorithm

This script detects a person's pose in an image and classifies whether their "left", "right", "both", or "None" arms are raised.

The algorithm relies on comparing the vertical coordinates of the wrists and shoulders.

1.  The script loads the pre-trained 'pose_detector.task' model from MediaPipe.
2.  It runs the pose landmarker on the input image, which returns a detection_result containing 33 landmarks for each detected person.
3. We get the 'NormalizedLandmark' objects for four specific points:
    landmarks[11] = Left Shoulder
    landmarks[12] = Right Shoulder
    landmarks[15] = Left Wrist
    landmarks[16] = Right Wrist
4.  For each arm, we check the '.visibility' score of both the wrist and shoulder. We only proceed if both are above a 'VISIBILITY_THRESHOLD'.
5. In image coordinates, 'y=0.0' is the top of the image. Therefore, an arm is considered "up" if its wrist's 'y' coordinate is less than its shoulder's 'y' coordinate.
    left_wrist.y < left_shoulder.y means the left arm is up.
    right_wrist.y < right_shoulder.y means the right arm is up.
6.  Based on the status of both arms, the script prints the final string: left, right, both, or None.