# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:13:00 2023

@author: ehsan
"""
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
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
from visualize_kmeans import *
from sklearn.cluster import DBSCAN
import re
import datetime
import pickle
import pandas as pd
import seaborn as sns
from skimage.io import imsave
import warnings
# Suppress low contrast warning
warnings.filterwarnings('ignore', category=UserWarning, module='skimage.io')



directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data_figs"
sample_files = [f for f in os.listdir(directory) if re.match(r'sample_\d+\.png', f)]
# Sort the sample_files list based on the numerical part of the filename
sample_files = sorted(sample_files, key=lambda x: int(re.search(r'(\d+)', x).group()))
final_features_list = []
output_directory=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data_figs_normalized"


for sample in sample_files:
    image_path = os.path.join(directory, sample)
    image = imread(image_path)
    # Convert image to grayscale
    gray_image = rgb2gray(image[:, :, :3])
    #grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    #only the first three channels are considered which are red,green,blue
    # Apply binary threshold
    thresh = threshold_otsu(gray_image)
    gray_image[gray_image == 1] = 0
    binary_image = gray_image > thresh
    dark_points, bright_points = classify_bright_points(gray_image, binary_image)
    bright_points_group1, bright_points_group2 = classify_points_based_on_intensity(bright_points, gray_image)
    dbscan_labels = apply_dbscan(bright_points_group1)
    bright_points_array = np.array(bright_points_group1)
    cluster_centers = compute_cluster_centers(bright_points_array, dbscan_labels)
    centroid_x = np.mean(cluster_centers[:, 0])
    centroid_y = np.mean(cluster_centers[:, 1])
    normalized_cluster_centers = cluster_centers - [centroid_x, centroid_y]
    sparse_img = points_to_image(normalized_cluster_centers.astype(int), gray_image.shape)
    # Generate an image with lines connecting every pair of cluster centers
    lines_img = create_lines_image(normalized_cluster_centers)
    # Visualization
    
    
    
    plt.imshow(image , cmap='gray')
    plt.show()
    
    
    plt.imshow(lines_img, cmap='gray')
    plt.title(f'Virtual Lines for Sample {sample[:-4]}')
    plt.axis('off')
    plt.show()
    save_path = os.path.join(output_directory, f'sample_{sample}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the figure after saving
    
    print(f'Figure for sample_{sample} saved at {save_path}')
   # Apply Hough Transform to the generated image
    h, theta, distances_hough = hough_line(lines_img)
    hough_data = get_hough_pairs(h, theta, distances_hough)
    check_for_nan(hough_data, "get_hough_pairs")
    hough_features = extract_hough_features(hough_data, N=30)
    check_for_nan(hough_features, "extract_hough_features")
    #combined_features = np.concatenate((cluster_features, hough_features))
    final_features_list.append(hough_features)
    print(f'Sample {sample[:-4]} completed!')


directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\features"
filename = "sample_1000.pkl"
full_path = os.path.join(directory, filename)
with open(full_path, 'wb') as file:
    pickle.dump(final_features_list, file)

print(f"List saved to {full_path}")