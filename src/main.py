import cv2
import math
from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
from clock_detection import ClockDetection 
from HandDetection import HandDetection

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()

    processor = ImageProcessor()
    detector = ClockDetection()
    hand_detector = HandDetection()

   
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

        if candidates : 
            #pick the best/largest candidate circle 
            x, y, radius = candidates[0]
            cv2.circle(display, (x, y), radius, (0, 255, 0), 3)
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            print(f"Hough Cicrle detected at center: ({x}, {y}), radius: {radius}")
            clock_found = True

            # Detect candidate lines
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