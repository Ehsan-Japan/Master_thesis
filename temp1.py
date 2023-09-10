# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 23:42:46 2023

@author: ehsan
"""

import os
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def classify_bright_points(gray_image, binary_image):
    """Classify bright points in the binary image based on their grayscale intensity."""
    white_points = np.argwhere(binary_image)
    intensities = np.array([gray_image[y, x] for y, x in white_points]).reshape(-1, 1)
    
    # Classify intensities using KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(intensities)
    labels = kmeans.labels_
    
    # Identify more bright and less bright points based on clustering
    more_bright_points = [point for point, label in zip(white_points, labels) if label == np.argmax(kmeans.cluster_centers_)]
    less_bright_points = [point for point, label in zip(white_points, labels) if label != np.argmax(kmeans.cluster_centers_)]
    
    return less_bright_points, more_bright_points

def compute_average_center(points):
    """Compute the average center of the given points."""
    return np.mean(points, axis=0)

# Load and preprocess image

sample = "sample_1.png"

directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data_figs"


# sample_files = [f for f in os.listdir(directory) if re.match(r'sample_\d+\.png', f)]
# # Sort the sample_files list based on the numerical part of the filename
# sample_files = sorted(sample_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

# image_path = os.path.join(directory, sample)
# image = imread(image_path)

# # Convert to grayscale and threshold
# gray_image = rgb2gray(image[:, :, :3])
# thresh = threshold_otsu(gray_image)
# binary_image = gray_image > thresh

# # Classify bright points
# less_bright, more_bright = classify_bright_points(gray_image, binary_image)

# # Compute the average center of the 'more bright' points
# center = compute_average_center(more_bright)

# # Visualize the results
# fig, ax = plt.subplots()
# ax.imshow(image)

# # Draw the more bright points and the connecting lines
# more_bright_np = np.array(more_bright)
# ax.scatter(more_bright_np[:, 1], more_bright_np[:, 0], c='yellow', s=10)
# ax.plot(center[1], center[0], 'ro')  # Plotting the average center as a red point

# plt.show()





def cluster_bright_points(gray_image, binary_image, n_clusters=2):
    """Cluster bright points and return their centroids."""
    white_points = np.argwhere(binary_image)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(white_points)
    centroids = kmeans.cluster_centers_
    
    return centroids, white_points, kmeans.labels_


def connect_centroids_with_lines(ax, centroids):
    """Draw lines connecting every centroid to every other centroid."""
    for i in range(centroids.shape[0]):
        for j in range(i+1, centroids.shape[0]):
            ax.plot([centroids[i, 1], centroids[j, 1]], 
                    [centroids[i, 0], centroids[j, 0]], 
                    color='blue')


image_path = os.path.join(directory, sample)
image = imread(image_path)

# Convert to grayscale and threshold
gray_image = rgb2gray(image[:, :, :3])
thresh = threshold_otsu(gray_image)
binary_image = gray_image > thresh

# 1. Get the indices (y, x) of the pixels with True labels in the binary image
bright_points_indices = np.argwhere(binary_image)

# 2. Extract the intensities of these pixels from the original image
# Convert the RGB image to grayscale for intensity extraction
intensities = gray_image[binary_image]

# 3. Cluster these intensities into two groups using k-means
kmeans = KMeans(n_clusters=2, random_state=0).fit(intensities.reshape(-1, 1))
labels = kmeans.labels_

# 4. Identify the brighter cluster
brighter_cluster_label = np.argmax(kmeans.cluster_centers_)

# Extracting the indices of the brighter points
brighter_points_indices = bright_points_indices[labels == brighter_cluster_label]
less_bright_points_indices = bright_points_indices[labels != brighter_cluster_label]

# Create a white background image with the same shape as the original
white_background = np.ones((image.shape[0], image.shape[1], 3))

# Set the brighter points to red
for y, x in brighter_points_indices:
    white_background[y, x] = [1, 0, 0]  # RGB for red

# Visualize the result
plt.figure(figsize=(8, 8))
plt.imshow(white_background)
plt.axis('off')
plt.title('Brightest Points')
plt.show()





