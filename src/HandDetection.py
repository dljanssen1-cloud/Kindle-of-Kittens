import cv2
from matplotlib.pyplot import gray
import numpy as np
import math


class HandDetection:
    def preprocess_hands(self, gray, use_clahe=False):
           # Optionally improve local contrast
            if use_clahe:
                gray = self.enhance_contrast(gray)

            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            edges = cv2.Canny(blurred, 20, 60)

            return edges

    def enhance_contrast(self, gray):
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        return enhanced
    
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
        min_length = radius * 0.20

        for x1, y1, x2, y2 in lines:

            # Calculate the distance of both endpoints to the center
            distance1 = math.hypot(x1 - center_x, y1 - center_y)
            distance2 = math.hypot(x2 - center_x, y2 - center_y)

            # Find the endpoint closest to the center
            closest_distance = min(distance1, distance2)

             # Calculate the length of the line
            length = math.hypot(x2 - x1, y2 - y1)

            # Only keep lines that start or end near the center
            if closest_distance <= max_distance and length >= min_length:
                filtered_lines.append((x1, y1, x2, y2))

        return filtered_lines

    def calculate_angle(self, line, center_x, center_y):

        x1, y1, x2, y2 = line

        # Calculate the distance of both endpoints from the center
        distance1 = math.hypot(x1 - center_x, y1 - center_y)
        distance2 = math.hypot(x2 - center_x, y2 - center_y)

        # Determine the direction of the line
        if distance1 > distance2:
            dx = x1 - x2
            dy = y1 - y2
        else:
            dx = x2 - x1
            dy = y2 - y1

        # Calculate the angle
        angle = math.degrees(math.atan2(dx, -dy)) % 360

        return angle

    def merge_similar_lines(self, lines, center_x, center_y):

        merged_lines = []
        angle_tolerance = 8

        # Sort lines from longest to shortest
        sorted_lines = sorted(
            lines,
            key=lambda line: math.hypot(
                line[2] - line[0],
                line[3] - line[1]
            ),
            reverse=True
        )

        for line in sorted_lines:

            angle = self.calculate_angle(line, center_x, center_y)

            similar = False

            for existing_line in merged_lines:

                existing_angle = self.calculate_angle(
                    existing_line, center_x, center_y
                )

                # Calculate the difference between the angles
                difference = abs(angle - existing_angle)
                difference = min(difference, 360 - difference)

                # Check whether the lines point in a similar direction
                if difference <= angle_tolerance:
                    similar = True
                    break

            # Only keep the line if no similar line already exists
            if not similar:
                merged_lines.append(line)

        return merged_lines