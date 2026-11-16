#Add all the preprocessing steps here, such as gray-scaling, normalization, blurring, etc.
import cv2

class ImageProcessor:
     def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)