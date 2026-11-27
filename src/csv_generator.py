import cv2
import pandas as pd

from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
from clock_detection import ClockDetection
from HandDetection import HandDetection
from time_reader import TimeReader


def generate_submission():

    # Load all test images
    loader = ImageLoader("../ivp-2026/test/test")
    images = loader.load_images()

    processor = ImageProcessor()
    detector = ClockDetection()
    hand_detector = HandDetection()
    time_reader = TimeReader()

    results = []

    for filename, image in images:

        print(f"Processing {filename}")

        # Preprocess the image
        image = processor.resize_image(image)
        gray = processor.to_grayscale(image)
        blurred = processor.apply_blur(gray)
        edge = processor.edge_detection(blurred)

        # Detect the clock face
        candidates = detector.detect_candidates(blurred)

        clock_found = False

        # Default prediction if detection fails
        hour, minute = 12, 0

        if candidates:

            x, y, radius = candidates[0]
            clock_found = True

        else:

            # Fallback to contour detection
            contour = processor.find_clock_contour(edge)

            if contour is not None:

                area = cv2.contourArea(contour)
                img_area = image.shape[0] * image.shape[1]

                if area > 0.15 * img_area:

                    center = processor.centerclock_detection(contour)

                    if center is not None:

                        x, y = center

                        _, _, w, h = cv2.boundingRect(contour)
                        radius = int(round(max(w, h) / 2))

                        clock_found = True

        if clock_found:

            # Transform the clock face into polar coordinates
            polar = hand_detector.polar_transform(gray, x, y, radius)

            # Calculate radial contrast scores
            scores = hand_detector.calculate_radial_scores(polar)

            # Detect potential clock hand directions
            peaks = hand_detector.detect_peaks(scores)

            # Merge peaks belonging to the same clock hand
            peaks = hand_detector.merge_polar_peaks(peaks)

            # Select the hour and minute hands
            hands = time_reader.pick_polar_hands(peaks, polar, radius, hand_detector)

            # Convert the angles into time
            if hands is not None:

                hour_angle, minute_angle = hands

                hour, minute = time_reader.angles_to_time(
                    hour_angle, minute_angle
                )

            else:
                print("No hands detected. Using fallback time.")

        else:
            print("No clock detected. Using fallback time.")

        # Format the predicted time
        predicted_time = f"{hour:02d}:{minute:02d}"

        print(f"Predicted time: {predicted_time}")

        # Store the prediction
        results.append({
            "ID": filename,
            "time": predicted_time
        })

    # Convert all predictions into a DataFrame
    df_preds = pd.DataFrame(results, columns=["ID", "time"])

    # Save the predictions
    output_path = "submission_polar.csv"
    df_preds.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(df_preds)}")


if __name__ == "__main__":
    generate_submission()