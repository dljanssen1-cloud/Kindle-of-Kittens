import cv2
import math
from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
from clock_detection import ClockDetection 
from HandDetection import HandDetection
from time_reader import TimeReader

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()

    processor = ImageProcessor()
    detector = ClockDetection()
    hand_detector = HandDetection()
    time_reader = TimeReader()

   
#load five images:
    for filename, image in images[:20]:
        print(f"Loaded {filename}")
        
        # Resize the image before processing
        image = processor.resize_image(image)

        gray = processor.to_grayscale(image)

        #Apply Gaussian blur 
        blurred = processor.apply_blur(gray)

         #edge detection 
        edge = processor.edge_detection(blurred)

        candidates = detector.detect_candidates(blurred)

        #display = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # Convert edge image to BGR for display
        display = image.copy()  # I use the original image see if the detected lines are correct
        clock_found = False 
        
        #keep the original colour image for display 
        #display = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        if candidates:

            x, y, radius = candidates[0]
            cv2.circle(display, (x, y), radius, (0, 255, 0), 3)
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            print(f"Hough Circle detected at center: ({x}, {y}), radius: {radius}")
            clock_found = True

            # Transform the clock face into polar coordinates
            polar = hand_detector.polar_transform(gray, x, y, radius)

            # Calculate radial contrast scores
            scores = hand_detector.calculate_radial_scores(polar)

            # Detect potential clock hand directions
            peaks = hand_detector.detect_peaks(scores)

            print(f"Detected {len(peaks)} polar peaks")

            # Merge peaks belonging to the same clock hand
            peaks = hand_detector.merge_polar_peaks(peaks)

            print(f"Remaining {len(peaks)} merged polar peaks")

            for angle, score in peaks:
                print(f"Polar angle: {angle:.2f}°, Score: {score:.2f}")

            # Visualise the strongest polar peaks
            for angle, score in peaks[:5]:

                angle_rad = math.radians(angle)

                length = radius * 0.75

                tip_x = int(x + length * math.sin(angle_rad))
                tip_y = int(y - length * math.cos(angle_rad))

                cv2.line(
                    display,
                    (x, y),
                    (tip_x, tip_y),
                    (255, 0, 255),
                    2
                )
            
            # Display the radial contrast scores
            # plt.figure(figsize=(10, 4))
            # plt.plot(angles, scores)
            # plt.xlabel("Clock angle (degrees)")
            # plt.ylabel("Radial contrast score")
            # plt.title("Possible Clock Hand Directions")
            # plt.xlim(0, 360)
            # plt.grid()
            # plt.show()

            #cv2.imshow("Polar Clock", polar)    

            # Existing hand detection
            hand_edges = hand_detector.preprocess_hands(gray)

            # Remove edges outside the clock face
            masked_edges = hand_detector.mask_clock(hand_edges, x, y, radius)

            lines = hand_detector.detect_candidate_lines(masked_edges, radius)

            print(f"Detected {len(lines)} candidate lines")

            # Filter unwanted lines
            filtered_lines = hand_detector.filter_unwanted_lines(
                lines, x, y, radius
            )

            print(f"Remaining {len(filtered_lines)} filtered lines")

            # Merge similar lines
            merged_lines = hand_detector.merge_similar_lines(
                filtered_lines, x, y
            )

            print(f"Remaining {len(merged_lines)} merged lines")
            for line in merged_lines:

                x1, y1, x2, y2 = line

                angle = hand_detector.calculate_angle(line, x, y)

                length = math.hypot(x2 - x1, y2 - y1)

                print(f"Angle: {angle:.2f}°, Length: {length:.2f} px")

            # Draw all detected candidate lines
            for x1, y1, x2, y2 in merged_lines:
                cv2.line(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

            #calculate angles + time on hand edges
            # Calculate time using Hough lines
            hough_hands = time_reader.pick_hands(merged_lines, x, y)

            if hough_hands is not None:

                hour_angle, minute_angle = hough_hands

                hour, minute = time_reader.angles_to_time(
                    hour_angle, minute_angle
                )

                print(f"Hough Time: {hour:02d}:{minute:02d}")

            else:
                print("No Hough time found")


            # Calculate time using polar peaks
            polar_hands = time_reader.pick_polar_hands(peaks, polar, radius, hand_detector)

            if polar_hands is not None:

                hour_angle, minute_angle = polar_hands

                hour, minute = time_reader.angles_to_time(
                    hour_angle, minute_angle
                )

                print(f"Polar Time: {hour:02d}:{minute:02d}")
                # Visualise the selected hour and minute hands
                for angle, length, colour in [
                    (hour_angle, radius * 0.50, (0, 165, 255)),
                    (minute_angle, radius * 0.75, (0, 255, 255))
                ]:

                    angle_rad = math.radians(angle)

                    tip_x = int(x + length * math.sin(angle_rad))
                    tip_y = int(y - length * math.cos(angle_rad))

                    cv2.line(
                        display,
                        (x, y),
                        (tip_x, tip_y),
                        colour,
                        3
                    )
            else:
                print("No polar time found")

        else:
            #fallback to contour detection with strict size filtering 
            contour = processor.find_clock_contour(edge)


            if contour is not None:
                area = cv2.contourArea(contour)
                img_area = image.shape[0] * image.shape[1]

                #ensure the contour is actually large enough to be a clock 
                if area > 0.15 * img_area: 
                    cv2.drawContours(display, [contour], -1, (0, 255, 0), 3)
                    center = processor.centerclock_detection(contour)
                    if center is not None:
                        print("Contour center:", center)
                        cv2.circle(display, center, 5, (0, 0, 255), -1)
                        clock_found = True

        if not clock_found: 
            print("No valid clock detected!")


        # Resize only for display
        height, width = display.shape[:2]
        scale = min(1000 / width, 700 / height)

        display = cv2.resize(
            display, None, fx=scale, fy=scale
        )

        cv2.imshow("Processed Clock Image", display)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break 


    cv2.destroyAllWindows()

        # Add the next image-processing steps here

    print(f"Successfully loaded {len(images)} images.")

if __name__ == "__main__":
    main()  