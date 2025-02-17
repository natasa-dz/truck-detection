# 🚚 Vehicle Counting using Background Subtraction

This Python script implements vehicle (specifically truck) counting using background subtraction techniques in video footage. It uses OpenCV to process and analyze video frames, detect moving objects (trucks), and count them based on object tracking and contour detection. The system evaluates the performance of the truck count predictions using the **Mean Absolute Error (MAE)** metric against the ground truth provided in a CSV file.

## 📋 Features:
- **Background Subtraction**: Uses the MOG2 or KNN background subtraction algorithms to detect moving vehicles in the video. 🏙️
- **Object Tracking**: Tracks objects across frames based on their position, size, and movement. 🚚
- **Truck Detection**: Filters detected objects based on size (width and height) to identify trucks. 🚛
- **Post-processing**: Applies Gaussian blur and CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast and noise reduction. 🔍
- **Error Simulation**: Simulates potential errors in detection by comparing predicted truck counts with actual counts from a CSV file. ⚠️
- **Evaluation**: Calculates **Mean Absolute Error (MAE)** between predicted and real truck counts. 📊

---
## 🛠️ Requirements:
- Python 3.x
- OpenCV (`opencv-python`, `opencv-contrib-python`) 🖥️
- NumPy 📐
- Pandas 📊
- scikit-learn 🧠

You can install the required libraries by running:
```bash
pip install opencv-python opencv-contrib-python numpy pandas scikit-learn
```
---
## 📝 Usage

1. Run the Script:
- Use the following command to run the script:
  ```bash
  python main.py <path_to_dataset_folder>
  ```
  The script will process all videos in the dataset folder and compare the predicted truck counts to the real counts in the CSV file. 📈

2. Results:
- The Mean Absolute Error (MAE) will be printed, indicating the performance of the vehicle counting model. 🏅


