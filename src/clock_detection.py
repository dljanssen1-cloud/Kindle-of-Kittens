import cv2
import numpy as np


class ClockDetection:

    def detect_candidates(self, image):

        height, width = image.shape[:2]
        min_dimension = min(height, width)

        # Start with strict detection and relax if necessary
        for threshold in [70, 55, 40, 30]:

            circles = cv2.HoughCircles(
                image,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(100, int(min_dimension * 0.20)),
                param1=100,
                param2=threshold,
                minRadius=int(min_dimension * 0.08),
                maxRadius=int(min_dimension * 0.48)
            )

            if circles is not None:

                circles = np.round(circles[0]).astype(int)

                candidates = []

                for x, y, radius in circles:
                    candidates.append(
                        (int(x), int(y), int(radius))
                    )

                return candidates

        return []

    def score_circle(self, edges, x, y, radius):

        height, width = edges.shape[:2]

        angles = np.linspace(
            0, 2 * np.pi, 360, endpoint=False
        )

        # Calculate points along the circumference
        xs = np.round(
            x + radius * np.cos(angles)
        ).astype(int)

        ys = np.round(
            y + radius * np.sin(angles)
        ).astype(int)

        # Check which points are inside the image
        valid = (
            (xs >= 0) & (xs < width) &
            (ys >= 0) & (ys < height)
        )

        if np.mean(valid) < 0.90:
            return 0

        # Check how many circumference points contain edges
        edge_values = edges[ys[valid], xs[valid]]

        score = np.mean(edge_values > 0)

        return score

    def select_best_circle(self, candidates, edges):

        if not candidates:
            return None

        # Allow small inaccuracies in the circle boundary
        kernel = np.ones((5, 5), dtype=np.uint8)
        expanded_edges = cv2.dilate(edges, kernel)

        best_circle = max(
            candidates,
            key=lambda circle: self.score_circle(
                expanded_edges,
                *circle
            )
        )

        return best_circle