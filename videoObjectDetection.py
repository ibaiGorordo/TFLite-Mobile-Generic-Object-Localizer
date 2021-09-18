import cv2
import pafy
from genericDetector import GenericDetector

model_path='models/object_detection_mobile_object_localizer_v1_1_default_1.tflite'
threshold = 0.25

out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (720,720))

# Initialize video
# cap = cv2.VideoCapture("video.avi")

videoUrl = 'https://youtu.be/uKyoV0uG9rQ'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)

# Initialize object detection model
detector = GenericDetector(model_path, threshold)

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
    except:
        continue

    if ret: 

         # Draw the detected objects
        detections = detector(frame)
        detection_img = detector.draw_detections(frame, detections)

        out.write(detection_img)
        cv2.imshow("Detections", detection_img)

    else:
        break

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()