from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
import cv2

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()
    processor = ImageProcessor()

    for filename, image in images:
        print(f"Loaded {filename}")
        gray = processor.to_grayscale(image)
        edge = processor.edge_detection(gray)

        cv2.imshow("Image", image)
        cv2.imshow("edge", edge)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


        # Add the next image-processing steps here

    print(f"Successfully loaded {len(images)} images.")

if __name__ == "__main__":
    main()