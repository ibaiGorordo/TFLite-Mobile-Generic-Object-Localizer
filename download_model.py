import requests

url = "https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1?lite-format=tflite"
contents = requests.get(url).content

with open("models/object_detection_mobile_object_localizer_v1_1_default_1.tflite" ,mode='wb') as f: # wb でバイト型を書き込める
	f.write(contents)