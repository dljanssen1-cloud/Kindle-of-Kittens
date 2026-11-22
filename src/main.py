import cv2
from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()

    processor = ImageProcessor()

   
#load five images:
    for filename, image in images[:5]:
        print(f"Loaded {filename}")

        gray = processor.to_grayscale(image)

        #Apply Gaussian blur 
        blurred = processor.apply_blur(gray)

        #edge detection 
        edge = processor.edge_detection(blurred)

        #find the clock contour
        contour = processor.find_clock_contour(edge)

        #keep the original colour image for display 
        display = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

        if contour is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 3) #draw shape around contour
            
            center = processor.centerclock_detection(contour)

            if center is not None:
                print("Contour center:", center)

                cv2.circle(display, center, 5, (0, 0, 255), -1)

            else: 
                print("No clock detected!")


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