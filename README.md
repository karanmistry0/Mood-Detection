# Mood Detection 

## Overview
This project implements an image classifier using TensorFlow. It involves loading and preprocessing image data, training a machine learning model, and evaluating its performance. The classifier is designed to recognize images and classify them into predefined categories.

## Features
- Removes corrupted or invalid images during preprocessing.
- Scales and normalizes image data for optimal model performance.
- Splits data into training and testing sets.
- Trains a deep learning model using TensorFlow.
- Evaluates and visualizes model performance.

## Libraries Used
- `tensorflow`: For building and training the image classification model.
- `numpy`: For numerical computations and data manipulation.
- `matplotlib`: For data visualization.
- `os`: For file and directory operations.
- `imghdr`: For image type validation.
- `cv2` (OpenCV): For advanced image processing.

## Data
- Download the dataset from this https://www.kaggle.com/datasets/sanidhyak/human-face-emotions

## How to Run
1. **Install Dependencies**:
   Ensure you have Python installed. Use the following command to install the required libraries:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python
   ```

2. **Load the Notebook**:
   Open the `ImageClassifier.ipynb` file in Jupyter Notebook or any compatible IDE.

3. **Run the Cells**:
   Execute each cell sequentially to preprocess data, train the model, and evaluate its performance.

4. **Customize the Model**:
   Modify the parameters or architecture in the code to experiment with different settings.

## Notes
- Ensure your dataset is organized into folders by class labels before running the notebook.
- The project includes steps to clean and validate the dataset, ensuring no corrupted images interfere with training.
- For optimal performance, consider using a GPU-enabled environment when training the model.

## Project Structure
- **Data Preprocessing**: Cleaning and preparing the dataset.
- **Model Training**: Building and training the TensorFlow model.
- **Evaluation**: Testing the model and visualizing results.

## Acknowledgments
This project uses TensorFlow for deep learning and OpenCV for preprocessing, demonstrating the power of these libraries in creating robust image classifiers.

