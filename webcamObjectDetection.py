import cv2

from genericDetector import GenericDetector

model_path='models/object_detection_mobile_object_localizer_v1_1_default_1.tflite'
threshold = 0.3

# Initialize object detection model
detector = GenericDetector(model_path, threshold)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()

    # Draw the detected objects
    detections = detector(frame)
    detection_img = detector.draw_detections(frame, detections)

    
    cv2.imshow("Detections", detection_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break