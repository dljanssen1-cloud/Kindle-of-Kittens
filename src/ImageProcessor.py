#Add all the preprocessing steps here, such as gray-scaling, normalization, blurring, etc.
import cv2
import math
import numpy as np

class ImageProcessor:

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def edge_detection(self, gray):
        blurred = cv2.medianBlur(gray, 5) #reduce noise while keep edges
        edges = cv2.Canny(blurred, 20, 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) #connects small gap in edges
        return closed

    def find_clock_contour(self, edges):
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #find all contours in the image

        best_contour = None
        largest_diameter = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = (4 * math.pi * area) / (perimeter ** 2) #how circular is the shape 1=perfect, 0=not at all
            x, y, w, h = cv2.boundingRect(contour) #calculate rectangle
            diameter = max(w, h) #choose largest dimension of the rectangle

            if circularity >= 0.3 and diameter > largest_diameter:
                largest_diameter = diameter
                best_contour = contour

        return best_contour

    def centerclock_detection(self, contour):
        if contour is None:
            return None

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            return None

        #calculate coordinate of x and y (moment of y/x / area)
        coordinate_x = int(moments["m10"] / moments["m00"])
        coordinate_y = int(moments["m01"] / moments["m00"])

        return coordinate_x, coordinate_y



     
    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image):
        return cv2.GaussianBlur(image, (9, 9), 2)
