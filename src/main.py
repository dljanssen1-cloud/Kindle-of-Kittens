import cv2
from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
from clock_detection import ClockDetection 

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()

    processor = ImageProcessor()
    detector = ClockDetection()

   
#load five images:
    for filename, image in images[:5]:
        print(f"Loaded {filename}")

        gray = processor.to_grayscale(image)

        #Apply Gaussian blur 
        blurred = processor.apply_blur(gray)

         #edge detection 
        edge = processor.edge_detection(blurred)

        candidates = detector.detect_candidates(blurred)

        display = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)  # Convert edge image to BGR for display
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