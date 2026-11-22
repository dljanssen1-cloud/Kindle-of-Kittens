from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
import cv2
import numpy as np

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()
    processor = ImageProcessor()

    for filename, image in images:
        print(f"Loaded {filename}")
        gray = processor.to_grayscale(image)
        edge = processor.edge_detection(gray)
        contour = processor.find_clock_contour(edge)

        contour_output = image.copy()

        if contour is not None:
            cv2.drawContours(contour_output, [contour], -1, (0, 255, 0), 3) #draw shape around contour
            center = processor.centerclock_detection(contour)

            if center is not None:
                print("Contour center:", center)
                cv2.circle(contour_output, center, 5, (0, 0, 255), -1)


        cv2.imshow("Contour detection + center", contour_output)
        cv2.imshow("Edge detection", edge)

        key = cv2.waitKey(0) #press key to get next image set
        if key == ord("q"):  #press q to quit
            break

    cv2.destroyAllWindows()

        # Add the next image-processing steps here

    print(f"Successfully loaded {len(images)} images.")

if __name__ == "__main__":
    main()