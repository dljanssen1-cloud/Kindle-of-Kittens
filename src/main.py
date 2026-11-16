from ImageLoader import ImageLoader


def main():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()

    for filename, image in images:
        print(f"Loaded {filename}, shape: {image.shape}")

        # Add the next image-processing steps here
    print(f"Successfully loaded {len(images)} images.")

if __name__ == "__main__":
    main()