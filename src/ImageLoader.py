from pathlib import Path
import cv2


class ImageLoader:
    def __init__(self, folder):
        self.folder = Path(folder)

    def load_images(self):
        images = []

        for path in sorted(self.folder.iterdir()):
            if path.suffix.lower() in {".jpg"}:
                image = cv2.imread(str(path))

                if image is None:
                    raise ValueError(f"Could not load image: {path}")

                images.append((path.name, image))

        return images