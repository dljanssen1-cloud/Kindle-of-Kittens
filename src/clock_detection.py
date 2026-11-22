
import cv2
import numpy as np


class ClockDetection:

    def detect_candidates(self, image):

        height, width = image.shape[:2]
        min_dimension = min(height, width)

        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(100, int(min_dimension * 0.20)),
            param1=100,
            param2=70,
            minRadius=int(min_dimension * 0.08),
            maxRadius=int(min_dimension * 0.48)
        )

        if circles is None:
            return []

        circles = np.round(circles[0]).astype(int)

        candidates = []

        for x, y, radius in circles:
            candidates.append((x, y, radius))

        return candidates
