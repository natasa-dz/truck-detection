import cv2 as cv
import numpy as np
import sys
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Korišćeni izvori pri radu:
# - https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
# - ChatGPT

def main(video_path, algo='MOG2'):

    if algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        return 0 
    

    truck_count = 0
    tracked_objects = []  # List to hold tracked objects (centroid and ID)

    # Min and max values for which consider an item as a truck
    min_width = 120
    min_height = 72

    max_width = 200  
    max_height = 170  

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        gaussian_blur = cv.GaussianBlur(frame, (9, 9), 10.0)

        # height, width = frame.shape[:2]
        # line_position = int(height * 0.8)  
        gaussian_blur = cv.GaussianBlur(frame, (9, 9), 10.0)
        sharpened = cv.addWeighted(frame, 1.5, gaussian_blur, -0.5, 0)

        gray = cv.cvtColor(sharpened, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        fgMask = backSub.apply(equalized)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel, iterations= 1)
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel, iterations = 5)


        # Find contours from the mask
        contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:  
            # Filter by size
            area = cv.contourArea(contour)
            if area < 1000:
                continue
            
            x, y, w, h = cv.boundingRect(contour)

            if min_width <= w <= max_width and min_height <= h <= max_height and h%2 !=0 and w%2 !=0:
                cx, cy = x + w // 2, y + h // 2

                match_found = False
                for obj in tracked_objects:
                    obj_id, prev_cx, prev_cy = obj

                    if abs(cx - prev_cx) < 40 and abs(cy - prev_cy) < 40:  # Close enough to match
                        match_found = True
                        break

                if not match_found:
                    truck_count += 1  # Increment the count for a new truck
                    tracked_objects.append((truck_count, cx, cy))  # Track this object
                    #print(f"Truck {truck_count}: Width = {w}, Height = {h},  Coordinates = ({cx}, {cy})")

                #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
                #cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red centroid point
                #cv.putText(frame, f'Truck {truck_count}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Yellow label
                
        # counting line
        #cv.line(frame, (0, line_position), (width, line_position), (255, 0, 0), 2)  # Blue line        
        #cv.putText(frame, f'Trucks Counted: {truck_count}', (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        #cv.imshow('Frame', frame)
        #cv.imshow('Foreground Mask', fgMask)
        
    
    # capture.release()
    # cv.destroyAllWindows()
    return truck_count

def evaluate_counts(predicted_counts, real_counts):
    mae = mean_absolute_error(real_counts, predicted_counts)
    print(mae)
    return mae

if __name__ == "__main__":

    dataset_folder = sys.argv[1]
    counts_file = f"{dataset_folder}/counts.csv" 

    counts_df = pd.read_csv(counts_file)
    real_counts = counts_df['count'].tolist()
    file_names = counts_df['file'].tolist()

    predicted_counts = []
    for video_file in file_names:
        video_path = f"{dataset_folder}/{video_file}"
        count = main(video_path)
        predicted_counts.append(count)
    
    evaluate_counts(predicted_counts, real_counts)
