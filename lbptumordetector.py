import cv2
import os
import tkinter as tk
from tkinter import filedialog
from skimage.feature import local_binary_pattern
import numpy as np

# Function to calculate the Local Binary Pattern (LBP) histogram of an image with adjusted parameters
def calculate_lbp(image, radius=3, num_points=24):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the LBP
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    # Calculate the histogram of the LBP image , decide how m
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to compute the histogram intersection distance between two histograms
def histogram_intersection(hist1, hist2):
    return cv2.compareHist(np.array(hist1, dtype=np.float32), np.array(hist2, dtype=np.float32), cv2.HISTCMP_INTERSECT)

# Function to detect the label of an uploaded image based on similarity with images in training folders using LBP
def detect_label(upload_image_path, training_folder_paths, threshold=0.5):
    upload_image = cv2.imread(upload_image_path)
    if upload_image is None:
        print("Error: Unable to read the uploaded image.")
        return "Unknown"

    # Calculate LBP histogram for the uploaded image
    upload_hist = calculate_lbp(upload_image)

    max_similarity = 0
    label = "Unknown"

    for folder_path in training_folder_paths:
        reference_images = os.listdir(folder_path)
        for image_name in reference_images:
            reference_image_path = os.path.join(folder_path, image_name)
            reference_image = cv2.imread(reference_image_path)
            if reference_image is None:
                print(f"Error: Unable to read the reference image {image_name} in folder {folder_name}.")
                continue
            # Calculate LBP histogram for the reference image
            reference_hist = calculate_lbp(reference_image)
            # Compute histogram intersection distance between the histograms
            similarity = histogram_intersection(upload_hist, reference_hist)
            if similarity > max_similarity:
                max_similarity = similarity
                label = os.path.basename(folder_path)

    if max_similarity > threshold:
        return label
    else:
        return "Unknown"

# Function to open a file dialog for image selection
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog
    return file_path

# Example usage
if __name__ == "__main__":
    training_folder_paths = [
        "Training/glioma",
        "Training/meningioma",
        "Training/notumor",
        "Training/pituitary"
    ]

    upload_image_path = select_image()
    if upload_image_path:
        result = detect_label(upload_image_path, training_folder_paths)
        
        if result != "Unknown":
            print("Classified label:", result)
        else:
            print("No match found in training data.")
    else:
        print("No image selected.")
