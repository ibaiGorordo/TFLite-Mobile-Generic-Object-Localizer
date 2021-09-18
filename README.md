# TFLite Mobile Generic Object Localizer
Python TFLite scripts for detecting objects of any class in an image without knowing their label. 

![TFLite Generic Object Localizer](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/output.jpg)
*Image taken from the OpenCV AI Kit - Lite, make sure to check it out: https://www.kickstarter.com/projects/opencv/opencv-ai-kit-oak-depth-camera-4k-cv-edge-object-detection*

### :exclamation::warning:The object **detector works better with images with few objects** and it starts to fail in more complex scenes. The model is suitable for automatically labelling objects for custom object detection models.

# Requirements

 * **OpenCV**, **imread-from-url** and **tensorflow==2.6.0 or tflite_runtime**. Also, **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

For the tflite runtime, you can either use tensorflow(make sure it is version 2.6.0 or above) `pip install tensorflow==2.6.0` or the [TensorFlow Runtime binary](https://github.com/PINTO0309/TensorflowLite-bin)

# tflite model
The original models was taken from [Tensorflow Hub](https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1), download it, and place it in the **[models folder](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/tree/main/models)**. 

# Original Tensorflow model
The Tensorflow model is also available in Tensorflow Hub and would be more suitable for computers: https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1
 
# Examples

 * **Image inference**:
 
 ```
 python imageObjectDetection.py 
 ```
 
 * **Webcam inference**:
 
 ```
 python webcamObjectDetection.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoObjectDetection.py
 ```

# Inference Examples
![Generic object detector figures](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/genericObjectLocalizer.gif)
 *Original video by Animist: https://youtu.be/uKyoV0uG9rQ*

## Cabybara detection
![Capybara TFLite detection](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/capybara.jpg)
 
*Original image: https://commons.wikimedia.org/wiki/File:Capybara_portrait.jpg*

## Coin detection
![Coin TFLite detection](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/coins.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Japanese_Coins.jpg*

## Shoe detection
![Shoe TFLite detection](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/sneakers.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Japanese_Coins.jpg*

## Spaceship detection
![Spaceship TFLite detection](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/spaceship.jpg)
 *Original image: https://en.wikipedia.org/wiki/Spacecraft#/media/File:SpaceX_Crew_Dragon_(More_cropped).jpg*

## Window detection
![Window TFLite detection](https://github.com/ibaiGorordo/TFLite-Generic-Mobile-Object-Localizer/blob/main/docs/img/window.jpg)
 *Original image: https://commons.wikimedia.org/wiki/File:Window_-_Paddington_-_London.JPG*

## And many more

# References:
* Original model: https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1

