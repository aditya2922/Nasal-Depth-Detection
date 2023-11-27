# Nasal Depth Detection for Malnourished Children

This project uses facial landmark detection to analyze the nasal depth of children, aiming to contribute to malnutrition detection.

## Project Overview

The goal of this project is to assess nasal depth as a potential indicator of malnutrition in children. Facial landmarks are detected using the dlib library, allowing for the calculation of distances and positions related to key points on the face.

## How it Works

1. **Facial Landmark Detection:**
   - Utilizes dlib's pre-trained facial landmark model (`shape_predictor_68_face_landmarks.dat`) for accurate detection of facial features.

2. **Distance Calculation:**
   - Defines specific points (e.g., 22, 23, 28) on the face and calculates distances between them, providing insights into nasal depth.

3. **Positional Analysis:**
   - Analyzes the position of the center point between the eyes and the center point of a triangle formed by specific facial landmarks.

4. **Visualization:**
   - Draws lines connecting the detected facial landmarks on the image for better visualization.

## Requirements

- Python
- OpenCV
- dlib

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/nasal-depth-detection.git
   cd nasal-depth-detection
    ```

2. **Download Facial Landmark Model**

To use this project, you need to download the `shape_predictor_68_face_landmarks.dat` model from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.

  Follow these steps:
  
  **Download Model:**
     - Visit [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
     - Download the `shape_predictor_68_face_landmarks.dat.bz2` file.
  
  **Extract Model:**
     - Extract the contents of the downloaded file. You should now have the `shape_predictor_68_face_landmarks.dat` model file.
  
  **Place in Project Directory:**
     - Move the extracted `shape_predictor_68_face_landmarks.dat` file to the project directory.
  
  Now you're ready to run the script with the facial landmark model properly set up.


## Usage

1. **Run the Script:**

    Execute the Python script in a Python environment.

    ```bash
    python nasal_depth_detection.py
    ```

2. **Input Image:**

    Enter the path of the input image when prompted.

3. **Analysis:**

    View the output image with lines connecting facial landmarks and receive information on distances and positions related to nasal depth.

## Contributing

Contributions to enhance the project or address any issues are welcome. Feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License.

## Acknowledgments

Thank you to the dlib library developers for providing a robust facial landmark detection model.

Explore the nasal depth detection project and contribute to advancing malnutrition detection methods for children!
