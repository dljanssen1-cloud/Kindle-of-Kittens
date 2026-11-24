import cv2
import numpy as np
import math


class HandDetection:
    def preprocess_hands(self, gray):

            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            edges = cv2.Canny(blurred, 20, 60)

            return edges
    
    def mask_clock(self, edges, center_x, center_y, radius):

            # Create a black mask with the same dimensions as the image
            mask = np.zeros(edges.shape[:2], dtype=np.uint8)

            # Draw a filled white circle inside the clock face
            cv2.circle(
                mask,
                (center_x, center_y),
                int(radius * 0.90),
                255,
                -1
            )

            # Keep only the edges inside the mask
            masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

            return masked_edges
    
    def detect_candidate_lines(self, edges, radius):

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=int(radius * 0.10),
            maxLineGap=20
        )

        if lines is None:
            return []

        candidates = []

        for line in lines:
            x1, y1, x2, y2 = line
            candidates.append((x1, y1, x2, y2))

        return candidates

    def filter_unwanted_lines(self, lines, center_x, center_y, radius):

        filtered_lines = []

        # Maximum allowed distance from the clock center
        max_distance = radius * 0.15

        for x1, y1, x2, y2 in lines:

            # Calculate the distance of both endpoints to the center
            distance1 = math.hypot(x1 - center_x, y1 - center_y)
            distance2 = math.hypot(x2 - center_x, y2 - center_y)

            # Find the endpoint closest to the center
            closest_distance = min(distance1, distance2)

            # Only keep lines that start or end near the center
            if closest_distance <= max_distance:
                filtered_lines.append((x1, y1, x2, y2))

        return filtered_lines