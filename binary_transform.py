# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:37:17 2023

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
from visualize_kmeans import *
from sklearn.cluster import DBSCAN
import re
import datetime
import pickle

def process_and_save_images(directory):
    # Get the list of sample files
    sample_files = [f for f in os.listdir(directory) if re.match(r'sample_\d+\.png', f)]
    # Sort the sample_files list based on the numerical part of the filename
    sample_files = sorted(sample_files, key=lambda x: int(re.search(r'(\d+)', x).group()))
    final_features_list = []
    
    for sample in sample_files:
        try:
            image_path = os.path.join(directory, sample)
            image = imread(image_path)
            # Convert image to grayscale
            gray_image = rgb2gray(image[:, :, :3])
            # Apply binary threshold
            thresh = threshold_otsu(gray_image)
            gray_image[gray_image == 1] = 0  # Make pink brighter (i.e., turn to zero)
            binary_image = gray_image > thresh
            
            # Visualization
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title("Original Image", fontsize=16)
            axes[0].axis('off')
            
            axes[1].imshow(gray_image, cmap="gray")
            axes[1].set_title("Grayscale Image", fontsize=16)
            axes[1].axis('off')
            
            axes[2].imshow(binary_image, cmap="gray")
            axes[2].set_title("Binary Image", fontsize=16)
            axes[2].axis('off')
            
            directory_fig = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\DBSCAN figs"
            save_path = os.path.join(directory_fig, sample)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        
            print(f'Sample {sample[:-4]} completed!')
            
        except Exception as e:
            print(f"Error processing sample {sample[:-4]}. Error: {e}")
            
            return final_features_list
            
directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data_figs"
final_features_list=process_and_save_images(directory)
    
    
    

    
  