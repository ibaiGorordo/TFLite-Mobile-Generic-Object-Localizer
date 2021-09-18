import cv2
from imread_from_url import imread_from_url

from genericDetector import GenericDetector


model_path='models/object_detection_mobile_object_localizer_v1_1_default_1.tflite'
threshold = 0.2

# Initialize object detection model
detector = GenericDetector(model_path, threshold)

# Read RGB image
image = imread_from_url("https://ksr-ugc.imgix.net/assets/034/889/438/46e41611066c0eeae3c25773e499e926_original.png?ixlib=rb-4.0.2&crop=faces&w=1024&h=576&fit=crop&v=1631721168&auto=format&frame=1&q=92&s=9ce81981923cea116129532639be5d37")

# Draw the detected objects
detections = detector(image)
detection_img = detector.draw_detections(image, detections)
    
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Detections", detection_img)
cv2.waitKey(0)

cv2.imwrite("output.jpg",detection_img)