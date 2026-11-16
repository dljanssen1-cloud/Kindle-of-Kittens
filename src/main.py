from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor

def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()
    processor = ImageProcessor()

    for filename, image in images:
        print(f"Loaded {filename}")
        gray = processor.to_grayscale(image)

        # Add the next image-processing steps here
    
    print(f"Successfully loaded {len(images)} images.")

if __name__ == "__main__":
    main()