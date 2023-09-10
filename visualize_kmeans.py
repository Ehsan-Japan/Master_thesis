# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 00:14:49 2023

@author: ehsan
"""

import os
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import erosion, dilation, rectangle
#from skimage.util import invert
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from scipy.stats import skew, kurtosis
from skimage.transform import hough_line
from skimage.morphology import dilation, disk
import cv2
import pickle
from skimage.draw import line_aa
from skimage.transform import resize
from skimage.draw import line


def cluster_with_kmeans_based_on_peaks(bright_points_group2, histogram_peaks):
    """
    Cluster the points in bright_points_group2 using KMeans based on the number of histogram peaks.

    Parameters:
    - bright_points_group2: Points to be clustered
    - histogram_peaks: Peaks identified in the histogram of pairwise distances

    Returns:
    - kmeans_labels: Cluster labels assigned by KMeans
    """
    
    # Number of clusters is based on the number of histogram peaks
    n_clusters = len(histogram_peaks)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_labels = kmeans.fit_predict(bright_points_group2)
    
    return kmeans_labels


def visualize_clustering_results(image, bright_points_group2, kmeans_labels):
    """
    Visualize the clustering results by overlaying the points on the original image.

    Parameters:
    - image: Original image
    - bright_points_group2: Points that were clustered
    - kmeans_labels: Cluster labels assigned to each point in bright_points_group2
    """
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    n_clusters = len(np.unique(kmeans_labels))
    
    for i in range(n_clusters):
        cluster_points = np.array(bright_points_group2)[kmeans_labels == i]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], label=f'Cluster {i}', edgecolors='k', s=50, alpha=0.7)
    
    plt.title("KMeans Clustering Results")
    plt.legend(loc="upper right")
    plt.axis('off')
    plt.show()

def get_kmeans_centers(bright_points_group2, kmeans_labels):
    """
    Retrieve the centers of clusters formed by KMeans.

    Parameters:
    - bright_points_group2: Points that were clustered
    - kmeans_labels: Cluster labels assigned to each point in bright_points_group2

    Returns:
    - kmeans_centers: Centers of the clusters
    """
    
    # Instantiate KMeans with the number of unique labels (clusters)
    kmeans = KMeans(n_clusters=len(np.unique(kmeans_labels)), random_state=0)
    
    # Fit KMeans to the data
    kmeans.fit(bright_points_group2)
    
    # Retrieve cluster centers
    kmeans_centers = kmeans.cluster_centers_
    
    return kmeans_centers



def visualize_cluster_centers(image, kmeans_centers):
    """
    Visualize only the cluster centers by overlaying them on the original image.

    Parameters:
    - image: Original image
    - kmeans_centers: Centers of the clusters
    """
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    
    for i, center in enumerate(kmeans_centers):
        plt.scatter(center[1], center[0], marker='X', color='red', s=100, label=f'Center {i}')
    
    plt.title("Cluster Centers on Image")
    plt.legend(loc="upper right")
    plt.axis('off')
    plt.show()


def classify_bright_points(gray_image, binary_image, intensity_threshold=None):
    """
    Classify the brightest points from the binary image into two groups based on 
    their intensity in the grayscale image.

    Parameters:
    - gray_image: Grayscale version of the image
    - binary_image: Binary version of the image after thresholding
    - intensity_threshold: Optional threshold to classify the bright points. If None, median is used.

    Returns:
    - bright_points_group1: Points with lower intensity
    - bright_points_group2: Points with higher intensity
    """
    
    # Identify all the white points in the binary image
    white_points = np.argwhere(binary_image)
    
    # Get intensities of these white points from the grayscale image
    intensities = np.array([gray_image[y, x] for y, x in white_points]).reshape(-1, 1)
    
    # If no specific threshold is provided, use KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(intensities)
    labels = kmeans.labels_
  
    # Classify the white points based on the cluster labels
    temp_group1 = [point for point, label in zip(white_points, labels) if label == 0]
    temp_group2 = [point for point, label in zip(white_points, labels) if label == 1]

    # Calculate the mean intensity for each group
    mean_intensity_group1 = np.mean([gray_image[y, x] for y, x in temp_group1])
    mean_intensity_group2 = np.mean([gray_image[y, x] for y, x in temp_group2])

    # Ensure bright_points_group2 is the brighter group
    if mean_intensity_group1 > mean_intensity_group2:
        bright_points_group1, bright_points_group2 = temp_group2, temp_group1
    else:
        bright_points_group1, bright_points_group2 = temp_group1, temp_group2
    
    return bright_points_group1, bright_points_group2



def classify_points_based_on_intensity(points, gray_image, intensity_threshold=None):
    """
    Classify points based on their intensity in the grayscale image.

    Parameters:
    - points: List of points to be classified
    - gray_image: Grayscale version of the image
    - intensity_threshold: Optional threshold to classify the points. If None, median is used.

    Returns:
    - group1: Points with lower intensity
    - group2: Points with higher intensity
    """
    
    # Get intensities of the points from the grayscale image
    intensities = np.array([gray_image[y, x] for y, x in points]).reshape(-1, 1)
    
    # If no specific threshold is provided, use KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(intensities)
    labels = kmeans.labels_
  
    # Classify the points based on the cluster labels
    temp_group1 = [point for point, label in zip(points, labels) if label == 0]
    temp_group2 = [point for point, label in zip(points, labels) if label == 1]

    # Calculate the mean intensity for each group
    mean_intensity_group1 = np.mean([gray_image[y, x] for y, x in temp_group1])
    mean_intensity_group2 = np.mean([gray_image[y, x] for y, x in temp_group2])

    # Ensure group2 is the brighter group
    if mean_intensity_group1 > mean_intensity_group2:
        group1, group2 = temp_group2, temp_group1
    else:
        group1, group2 = temp_group1, temp_group2
    
    return group1, group2

def plot_distance_histogram(bright_points_group2):
    """
    Plot a histogram of the pairwise distances for bright_points_group2.

    Parameters:
    - bright_points_group2: Points for which the distance matrix will be computed

    Returns:
    - dist_array: Flattened distance matrix (1D array of pairwise distances)
    """
    
    # Calculate pairwise distances
    distances = distance_matrix(bright_points_group2, bright_points_group2)
    
    # Flatten the matrix to get all pairwise distances in a 1D array
    # We only take the upper triangle to avoid duplicate distances (since the matrix is symmetric)
    dist_array = distances[np.triu_indices(distances.shape[0], k=1)]
    
    
    return dist_array


def find_distance_peaks(dist_array):
    """
    Find peaks in the histogram of pairwise distances without any height threshold.

    Parameters:
    - dist_array: 1D array of pairwise distances

    Returns:
    - peaks: Indices of the peaks in the histogram
    - properties: Properties of the peaks
    """
    
    # Compute histogram
    counts, bin_edges = np.histogram(dist_array, bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks without height threshold
    peaks, properties = find_peaks(counts)
    return peaks, properties,bin_centers,counts


def dbscan_clustering(bright_points_group2, radius_threshold, min_samples=5):
    """
    Apply DBSCAN clustering based on a specified radius or distance threshold.

    Parameters:
    - bright_points_group2: List of points to be considered.
    - radius_threshold: User-defined distance threshold for DBSCAN.
    - min_samples: The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - labels: Cluster labels for each point. -1 indicates noise (outliers).
    """
    
    # Convert list of points to a numpy array
    points_array = np.array(bright_points_group2)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=radius_threshold, min_samples=min_samples).fit(points_array)
    
    return clustering.labels_

def categorize_distances_kmeans(dist_array, n_clusters=3):
    """
    Categorize distances using KMeans clustering.

    Parameters:
    - dist_array: The array of distances.
    - n_clusters: Number of clusters to categorize into (default is 3).

    Returns:
    - labels: The label of each distance indicating its category.
    - centers: The center of each category.
    """
    
    # Reshape the data to fit the KMeans input requirement
    distances_reshaped = dist_array.reshape(-1, 1)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distances_reshaped)
    
    # Return the labels and the cluster centers
    return kmeans.labels_, kmeans.cluster_centers_



    
def apply_dbscan(points, eps_value=10, min_samples=5):
    """
    Apply DBSCAN clustering to the given points.

    Parameters:
    - points: Array of points to be clustered.
    - eps_value: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - dbscan_labels: Cluster labels for each point. -1 indicates noise.
    """
    
    clustering = DBSCAN(eps=eps_value, min_samples=min_samples).fit(points)
    return clustering.labels_

def pairwise_distances(centers):
    """Compute pairwise distances between cluster centers."""
    return np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)


def refine_bright_points(gray_image, bright_points_group2):
    """
    Refine the bright points by applying another intensity threshold.

    Parameters:
    - gray_image: Grayscale version of the image
    - bright_points_group2: Initial set of bright points

    Returns:
    - refined_points: Refined set of bright points after applying the threshold
    """
    
    # Convert the list to numpy array for easier indexing
    bright_points_array = np.array(bright_points_group2)
    
    # Get the intensities of the bright points
    intensities = gray_image[bright_points_array[:, 0], bright_points_array[:, 1]]
    
    # Determine an intensity threshold (using Otsu's method here)
    thresh = threshold_otsu(intensities)
    
    # Retain only those points that exceed the threshold
    refined_points = bright_points_array[intensities > thresh]
    
    return refined_points


def compute_cluster_centers(points, labels):
    """Compute the center of each cluster."""
    unique_labels = np.unique(labels)
    cluster_centers = []

    for label in unique_labels:
        if label == -1:  # Exclude noise points
            continue
        cluster_points = points[labels == label]
        center = cluster_points.mean(axis=0)
        cluster_centers.append(center)

    return np.array(cluster_centers)


def compute_aggregated_features(normalized_cluster_centers):
    # Compute aggregate features for the x-coordinates
    mean_x = np.mean(normalized_cluster_centers[:, 0])
    variance_x = np.var(normalized_cluster_centers[:, 0])
    skewness_x = skew(normalized_cluster_centers[:, 0])
    kurt_x = kurtosis(normalized_cluster_centers[:, 0])

    # Compute aggregate features for the y-coordinates
    mean_y = np.mean(normalized_cluster_centers[:, 1])
    variance_y = np.var(normalized_cluster_centers[:, 1])
    skewness_y = skew(normalized_cluster_centers[:, 1])
    kurt_y = kurtosis(normalized_cluster_centers[:, 1])

    # Combine and return these features
    features = [mean_x, variance_x, skewness_x, kurt_x, mean_y, variance_y, skewness_y, kurt_y]
    return features


def extract_hough_features(hough_result, N, angle_bins=5, distance_bins=5, variance_threshold=1e-7):
    """
    Extract fixed-length features from Hough Transform result.
    
    Parameters:
    - hough_result: A 2D array where each row corresponds to a detected line and columns are [distance, angle].
    - N: Maximum number of lines to consider.
    - angle_bins: Number of bins for angle histogram.
    - distance_bins: Number of bins for distance histogram.
    - variance_threshold: Threshold below which variance is considered almost zero.
    
    Returns:
    - features: A fixed-length feature vector.
    """
    
    # If there are fewer lines than N, pad with zeros
    if hough_result.shape[0] < N:
        padding = np.zeros((N - hough_result.shape[0], 2))
        hough_result = np.vstack((hough_result, padding))
    
    # If there are more lines than N, consider only the top N
    elif hough_result.shape[0] > N:
        hough_result = hough_result[:N]
    
    # Extract distances and angles
    distances = hough_result[:, 0]
    angles = hough_result[:, 1]
    
    # Normalize distances and angles if variance is very low
    if np.var(distances) < variance_threshold:
        distances = np.zeros_like(distances)
    if np.var(angles) < variance_threshold:
        angles = np.zeros_like(angles)
    
    # Compute aggregate features
    # mean_distance = np.mean(distances)
    # variance_distance = np.var(distances)
    # skewness_distance = 0 if np.var(distances) < variance_threshold else skew(distances)
    # kurt_distance = 0 if np.var(distances) < variance_threshold else kurtosis(distances)
    
    # mean_angle = np.mean(angles)
    # variance_angle = np.var(angles)
    # skewness_angle = 0 if np.var(angles) < variance_threshold else skew(angles)
    # kurt_angle = 0 if np.var(angles) < variance_threshold else kurtosis(angles)
    
    # Compute histograms
    #distance_hist, _ = np.histogram(distances, bins=distance_bins)
    #angle_hist, _ = np.histogram(angles, bins=angle_bins)
    
    # Concatenate all features
    features = np.concatenate([
        distances,  # Top N distances
        angles,  # Top N angles
    ])
    
    return features


def get_hough_pairs(h, theta, distances):
    """
    For each distance, find the angle with the highest intensity in the Hough space.
    
    Parameters:
    - h: Hough transform result.
    - theta: Array of angles.
    - distances: Array of distances.
    
    Returns:
    - hough_pairs: Array of shape (num_distances, 2) where each row is [distance, corresponding_angle].
    """
    
    # For each distance, find the angle with the highest intensity in the Hough space
    max_intensity_angles = theta[np.argmax(h, axis=1)]
    
    # Pair each distance with its corresponding angle
    hough_pairs = np.column_stack((distances, max_intensity_angles))
    
    return hough_pairs


def create_binary_from_coordinates(coords, shape):
    """
    Create a binary image from a list of coordinates.
    
    Parameters:
    - coords: List of coordinates.
    - shape: Shape of the output binary image.
    
    Returns:
    - binary_image: A binary image where pixels at given coordinates are set to True.
    """
    
    binary_image = np.zeros(shape, dtype=bool)
    for coord in coords:
        x, y = coord
        if 0 <= x < shape[0] and 0 <= y < shape[1]:  # Check bounds
            binary_image[int(x), int(y)] = True
    return binary_image


def cartesian_to_polar(coords):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Parameters:
    - coords: Array of shape (N, 2) where each row is [x, y].
    
    Returns:
    - polar_coords: Array of shape (N, 2) where each row is [r, theta].
    """
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    polar_coords = np.column_stack((r, theta))
    
    return polar_coords


def compute_aggregated_features_1d(data):
    """
    Compute aggregated features (mean, variance, skewness, kurtosis) for a 1D data array.
    
    Parameters:
    - data: 1D numpy array.
    
    Returns:
    - features: List of aggregated features.
    """
    mean_val = np.mean(data)
    variance_val = np.var(data)
    skewness_val = skew(data)
    kurt_val = kurtosis(data)
    
    return [mean_val, variance_val, skewness_val, kurt_val]

def points_to_image(points, img_shape):
    """
    Convert a set of 2D points to a binary image.
    
    Parameters:
    - points: Array of shape (N, 2) where each row is [y, x].
    - img_shape: Tuple (height, width) indicating the shape of the desired output image.
    
    Returns:
    - Binary image of shape img_shape.
    """
    image = np.zeros(img_shape, dtype=np.uint8)
    for point in points:
        if 0 <= point[0] < img_shape[0] and 0 <= point[1] < img_shape[1]:
            image[int(point[0]), int(point[1])] = 1
    return image



def points_to_thicker_image(points, img_shape, radius=3):
    """
    Convert a set of 2D points to a binary image with thicker points.
    
    Parameters:
    - points: Array of shape (N, 2) where each row is [y, x].
    - img_shape: Tuple (height, width) indicating the shape of the desired output image.
    - radius: The radius for dilation to increase point size.
    
    Returns:
    - Binary image of shape img_shape with thicker points.
    """
    image = np.zeros(img_shape, dtype=np.uint8)
    for point in points:
        if 0 <= point[0] < img_shape[0] and 0 <= point[1] < img_shape[1]:
            image[int(point[0]), int(point[1])] = 1
    # Use dilation to increase the size of the points
    return dilation(image, disk(radius))

def visualize_thicker_image(points, img_shape, radius=3):
    """
    Generate a sparse binary image with thicker points from the given points and visualize it.
    
    Parameters:
    - points: Array of shape (N, 2) where each row is [y, x].
    - img_shape: Tuple (height, width) indicating the shape of the desired output image.
    - radius: The radius for dilation to increase point size.
    """
    # Convert points to a thicker image
    thicker_img = points_to_thicker_image(points, img_shape, radius)
    
    # Display the image
    plt.imshow(thicker_img, cmap='gray')
    plt.title('Sparse Image with Thicker Points')
    plt.axis('off')
    plt.show()
    
    
def check_for_nan(array, step_name):
    """
    Check if there are NaN values in the array and print a message if found.
    
    Parameters:
    - array: The array to check for NaN values.
    - step_name: A string indicating the name of the step or operation.
    """
    if np.isnan(array).any():
        print(f"NaN values found during step: {step_name}")


def visualize_hough_lines_modified(img, h, theta, distances, num_lines=10):
    """
    Visualize the Hough lines on the image.
    
    Parameters:
    - img: The image on which to overlay the lines.
    - h: Hough transform accumulator.
    - theta: Array of angles (radians) used in the Hough transform.
    - distances: Array of distances used in the Hough transform.
    - num_lines: Number of strongest lines to display.
    
    Returns:
    - Visualization of the Hough lines overlaying the image.
    """
    
    # Find the strongest lines
    indices = np.argsort(h.ravel())[-num_lines:]
    strongest_distances = distances[indices // theta.shape[0]]
    strongest_theta = theta[indices % theta.shape[0]]
    
    # Create an output image to draw on and visualize the output
    out_img = np.zeros(img.shape, dtype=np.uint8)
    
    # For each line, convert polar coordinates to Cartesian and draw the line
    for dist, angle in zip(strongest_distances, strongest_theta):
        y0 = dist * np.sin(angle)
        x0 = dist * np.cos(angle)
        
        # Define start and end points for the line
        x1 = int(x0 + 1000*(-np.sin(angle)))
        y1 = int(y0 + 1000*(np.cos(angle)))
        x2 = int(x0 - 1000*(-np.sin(angle)))
        y2 = int(y0 - 1000*(np.cos(angle)))
        
        # Draw the line on the output image
        cv2.line(out_img, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Drawing in white with thickness 2
    
    return out_img


def plot_hough_and_cluster_centers(sparse_img, h, theta, distances, normalized_cluster_centers, num_lines=10):
    # First, let's display the sparse image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sparse_img, cmap='gray')
    
    # Overlay the normalized cluster centers
    ax.scatter(normalized_cluster_centers[:, 0], normalized_cluster_centers[:, 1], color='red', s=100, marker='o')
    
    # Get the number of lines to be drawn
    num_lines = min(num_lines, len(distances))
    
    # Sort distances by magnitude
    indices = np.argsort(distances)[:num_lines]
    
    for index in indices:
        # Get distance and angle for each line
        distance = distances[index]
        angle = theta[index]

        # Compute start and end points of the line
        y0 = (distance - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (distance - sparse_img.shape[1] * np.cos(angle)) / np.sin(angle)

        # Draw the line
        ax.plot([0, sparse_img.shape[1]], [y0, y1], '-r')

    ax.set_xlim(0, sparse_img.shape[1])
    ax.set_ylim(sparse_img.shape[0], 0)
    plt.show()
    
    
def load_data_from_path(data_path, label_path):
    with open(data_path, 'rb') as data_file, open(label_path, 'rb') as label_file:
        data = pickle.load(data_file)
        label = pickle.load(label_file)
    return data, label

def extract_features_from_data(data):
    hough_data = data['hough_data']
    distance_features = [d for d in hough_data["distance"]]
    return distance_features

def extract_target_from_data(data):
    c_matrix = data['data']['c']
    ccs_matrix = data['data']['ccs']
    freq = data['data']['freq']
    return c_matrix[0] + c_matrix[1] + ccs_matrix[0] + ccs_matrix[1] + [freq]    



def plot_feature_histograms(features):
    """
    Plot histograms for each feature.
    """
    num_features = len(features[0])
    
    for i in range(num_features):
        feature_values = [sample[i] for sample in features]
        plt.hist(feature_values, bins=30, edgecolor='k', alpha=0.7)
        plt.title(f"Feature {i} Distribution")
        plt.xlabel(f"Value of Feature {i}")
        plt.ylabel("Number of Samples")
        plt.show()


# def create_lines_image(cluster_centers, shape):
#     """
#     Create an image of the given shape with lines connecting every pair of cluster centers.
    
#     Args:
#     - cluster_centers (list of tuple): List of coordinates for cluster centers.
#     - shape (tuple): Shape of the output image.
    
#     Returns:
#     - ndarray: An image with lines connecting every pair of cluster centers.
#     """
#     # Initialize a blank image
#     img = np.zeros(shape, dtype=np.uint8)
    
#     # Draw lines between every pair of cluster centers
#     for i in range(len(cluster_centers)):
#         for j in range(i+1, len(cluster_centers)):
#             rr, cc, val = line_aa(int(cluster_centers[i][0]), int(cluster_centers[i][1]), 
#                                   int(cluster_centers[j][0]), int(cluster_centers[j][1]))
#             img[rr, cc] = val * 255  # Convert anti-aliased values to integers
            
#     return img    


def create_lines_image(points):
    """
    Given an array of 2D points, create an image where lines are drawn between every pair of points.
    
    Parameters:
    - points: Array of shape (N, 2) where each row is [y, x].
    
    Returns:
    - Image with dynamically sized shape based on points.
    """
    
    # Dynamically determine image size
    y_max, x_max = np.ceil(points.max(axis=0)).astype(int)
    y_min, x_min = np.floor(points.min(axis=0)).astype(int)
    img_shape = (y_max - y_min + 1, x_max - x_min + 1)
    
    image = np.zeros(img_shape, dtype=np.uint8)
    
    # Adjust points to the new coordinate system
    adjusted_points = points - [y_min, x_min]
    
    for i in range(len(adjusted_points)):
        for j in range(i+1, len(adjusted_points)):
            rr, cc = line(int(adjusted_points[i][0]), int(adjusted_points[i][1]), 
                          int(adjusted_points[j][0]), int(adjusted_points[j][1]))
            image[rr, cc] = 1
            
    return image