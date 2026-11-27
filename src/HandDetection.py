import cv2
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

        for line in lines.reshape(-1, 4):
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

    def polar_transform(self, gray, center_x, center_y, radius):

        # Number of radial and angular samples
        radial_samples = 300
        angular_samples = 720

        # Transform the circular clock face into a rectangular image
        polar = cv2.warpPolar(
            gray,
            (radial_samples, angular_samples),
            (float(center_x), float(center_y)),
            float(radius),
            cv2.WARP_POLAR_LINEAR
            | cv2.WARP_FILL_OUTLIERS
            | cv2.INTER_LINEAR
        )

        # Shift the polar image so that 0 degrees represents 12 o'clock
        polar = np.roll(polar, angular_samples // 4, axis=0)

        return polar

    def merge_polar_peaks(self, peaks, angle_threshold=8):

        # Sort peaks from strongest to weakest
        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)

        merged = []

        for angle, score in peaks:

            duplicate = False

            for existing_angle, existing_score in merged:

                # Calculate the smallest angular difference
                difference = abs(angle - existing_angle)
                difference = min(difference, 360 - difference)

                if difference < angle_threshold:
                    duplicate = True
                    break

            # Keep the peak if it represents a new direction
            if not duplicate:
                merged.append((angle, score))

        return merged

    def calculate_radial_scores(self, polar):

        polar = polar.astype(np.float32)

        radius = polar.shape[1]

        # Select different radial regions
        inner = polar[
            :, int(radius * 0.12):int(radius * 0.40)
        ]

        middle = polar[
            :, int(radius * 0.40):int(radius * 0.65)
        ]

        # Compare each direction with nearby directions
        inner_contrast = np.abs(
            inner -
            (
                np.roll(inner, 6, axis=0) +
                np.roll(inner, -6, axis=0)
            ) / 2
        )

        middle_contrast = np.abs(
            middle -
            (
                np.roll(middle, 6, axis=0) +
                np.roll(middle, -6, axis=0)
            ) / 2
        )

        inner_scores = np.mean(inner_contrast, axis=1)
        middle_scores = np.mean(middle_contrast, axis=1)

        # Give more importance to structures near the centre
        scores = (
            0.7 * inner_scores +
            0.3 * middle_scores
        )

        return scores

    def detect_peaks(self, scores):

        # Smooth the scores to reduce small fluctuations
        padded = np.pad(scores, (2, 2), mode="wrap")

        smoothed = np.convolve(
            padded,
            np.ones(5) / 5,
            mode="valid"
        )

        # Calculate an adaptive threshold
        threshold = np.mean(smoothed) + np.std(smoothed)

        peaks = []

        for i in range(len(smoothed)):

            previous = smoothed[(i - 1) % len(smoothed)]
            current = smoothed[i]
            following = smoothed[(i + 1) % len(smoothed)]

            # Check whether the current score is a local maximum
            if current > previous and current >= following:
                if current >= threshold:

                    angle = i * 360 / len(smoothed)

                    peaks.append((angle, current))

        # Sort peaks from strongest to weakest
        peaks.sort(key=lambda peak: peak[1], reverse=True)

        return peaks

    def estimate_polar_length(self, polar, angle, radius):

        polar = polar.astype(np.float32)

        angular_samples, radial_samples = polar.shape

        # Convert the angle into a row index
        row = int(round(angle * angular_samples / 360))
        row = row % angular_samples

        # Get the pixel values along the candidate direction
        hand = polar[row]

        # Get the pixel values beside the candidate direction
        left = polar[(row - 6) % angular_samples]
        right = polar[(row + 6) % angular_samples]

        # Estimate the surrounding background
        background = (left + right) / 2

        # Calculate the contrast along the candidate hand
        contrast = np.abs(hand - background)

        # Smooth the contrast values
        smoothed = np.convolve(
            contrast,
            np.ones(7) / 7,
            mode="same"
        )

        # Ignore the central pivot and outer clock markings
        start = int(radial_samples * 0.12)
        end = int(radial_samples * 0.85)

        region = smoothed[start:end]

        # Calculate an adaptive contrast threshold
        threshold = max(5, np.max(region) * 0.25)

        # Find positions with sufficient contrast
        valid_positions = np.where(region >= threshold)[0]

        if len(valid_positions) == 0:
            return 0.0

        # Estimate the tip of the clock hand
        tip = valid_positions[-1] + start

        # Convert the radial position back into pixels
        length = tip * radius / radial_samples

        return length