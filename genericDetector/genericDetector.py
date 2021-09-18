import time
import cv2
import numpy as np
from imread_from_url import imread_from_url

np.random.seed(0)
np.random.random(9)
np.random.random(15)
np.random.random(2021)

colors = np.random.randint(255, size=(100, 3), dtype=int)

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter

class GenericDetector():

    def __init__(self, model_path, threshold = 0.2):

        self.threshold = threshold

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image):

        return self.detect_objects(image)

    def initialize_model(self, model_path):

        self.interpreter = Interpreter(model_path=model_path, num_threads = 4)
        self.interpreter.allocate_tensors()

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        self.inference(input_tensor)

        # Process output data
        detections = self.process_output()

        return detections

    def prepare_input(self, image):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = cv2.resize(img, (self.input_width,self.input_height))
        input_tensor = input_tensor[np.newaxis,:,:,:]      

        return input_tensor

    def inference(self, input_tensor):
        # Peform inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

    def process_output(self):  

        # Get all output details
        boxes = self.get_output_tensor(0)
        classes = self.get_output_tensor(1)
        scores = self.get_output_tensor(2)
        num_objects = int(self.get_output_tensor(3))

        results = []
        for i in range(num_objects):
            if scores[i] >= self.threshold:
                result = {
                  'bounding_box': boxes[i],
                  'class_id': classes[i],
                  'score': scores[i]
                }
                results.append(result)
        return results

    def get_output_tensor(self, index):

        tensor = np.squeeze(self.interpreter.get_tensor(self.output_details[index]['index']))
        return tensor

    def getModel_input_details(self):

        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

    def getModel_output_details(self):
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def draw_detections(image, detections):

        img_height, img_width, _ = image.shape

        for idx, detection in enumerate(detections):
            box = detection['bounding_box']
            y1 = (img_height * box[0]).astype(int)
            y2 = (img_height * box[2]).astype(int)
            x1 = (img_width * box[1]).astype(int)
            x2 = (img_width * box[3]).astype(int)

            cv2.rectangle(image, (x1, y1), (x2, y2), (int(colors[idx,0]), int(colors[idx,1]), int(colors[idx,2])), 5)

        return image


if __name__ == '__main__':

    model_path='models/object_detection_mobile_object_localizer_v1_1_default_1.tflite'
    threshold = 0.2

    image = imread_from_url("https://ksr-ugc.imgix.net/assets/034/889/438/46e41611066c0eeae3c25773e499e926_original.png?ixlib=rb-4.0.2&crop=faces&w=1024&h=576&fit=crop&v=1631721168&auto=format&frame=1&q=92&s=9ce81981923cea116129532639be5d37")

    detector = GenericDetector(model_path, threshold)

    detections = detector(image)
    detection_img = detector.draw_detections(image, detections)
    
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.imshow("Detections", detection_img)
    cv2.waitKey(0)