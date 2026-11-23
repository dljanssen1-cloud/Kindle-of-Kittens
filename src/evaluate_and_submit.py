import cv2 
import pandas as pd 
import numpy as np 
from pathlib import Path 
from ImageLoader import ImageLoader
from ImageProcessor import ImageProcessor
from clock_detection import ClockDetection

def evaluate_and_generate():
    loader = ImageLoader("../ivp-2026/train/train")
    images = loader.load_images()

    processor = ImageProcessor() 
    detector = ClockDetection() 

    results = []

    for filename, image in images: 
        gray = processor.to_grayscale(image)
        blurred = processor.apply_blur(gray)
        edge = processor.edge_detection(blurred)

        candidates = detector.detect_candidates(blurred)

        center_x, center_y, radius = 0, 0, 0 

        if candidates: 
            #handle rounding for edge cases / coordinates 
            x, y, r = candidates[0]
            center_x, center_y, radius = int(round(x)), int(round(y)), int(round(r))
        else:
            contour = processor.find_clock_contour(edge)
            if contour is not None: 
                center = processor.centerclock_detection(contour)
                if center is not None: 
                    center_x, center_y = int(round(center[0])), int(round(center[1]))
                    #estimate radius from bounding box width/height 
                    _, _, w, h = cv2.boundingRect(contour)
                    radius = int(round(max(w, h)/2))


        #store prediction row matching submission format 
        results.append({
            "filename": filename,
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius
        })

        #convert to DataFrame 
        df_preds = pd.DataFrame(results)

        #save predictions to CSV 
        output_path = "predictions_output.csv"
        df_preds.to_csv(output_path, index=False)
        print(f"Predictions successfully generated and saved to {output_path}")

        #optional evaluation if ground truth is available
        train_csv_path = "../ivp-2026/train/train.csv" 
        if Path(train_csv_path).exists():
            df_true = pd.read_csv(train_csv_path)
            merged = pd.merge(df_preds, df_true, on="filename", suffixes=("_pred", "_true"))

            #calculate mean absolute error for center coordinates 
            mae_x = np.mean(np.abs(merged["center_x_pred"] - merged["center_x_true"]))
            mae_y = np.mean(np.abs(merged["center_y_pred"] - merged["center_y_true"]))

            print(f"Evaluation Results:")
            print(f"MAE X: {mae_x:.2f} pixels")
            print(f"MAE Y: {mae_y:.2f} pixels")

if __name__ == "__main__":
    evaluate_and_generate()
