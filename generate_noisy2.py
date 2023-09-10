# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:18:57 2023

@author: ehsan
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import stability as stab
import matplotlib.pyplot as plt
import itertools
from scipy.signal import convolve
import os
import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
import json
import pickle
import random
from preprocess import *
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
import datetime


def highlight_top_n(matrix, n=2):
    # Get the top n values
    top_values = np.sort(np.unique(matrix))[-n:]
    
    # Create a highlighted matrix
    highlighted_matrix = np.zeros_like(matrix)
    for value in top_values:
        highlighted_matrix[np.where(matrix == value)] = value
        
    return highlighted_matrix


def max_pooling(input_matrix, pool_size):
    # Define dimensions of the input matrix
    H, W = input_matrix.shape
    
    # Calculate output dimensions
    H_out = H // pool_size
    W_out = W // pool_size
    
    # Initialize output matrix
    output = np.zeros((H_out, W_out))
    
    for i in range(0, H, pool_size):
        for j in range(0, W, pool_size):
            output[i // pool_size, j // pool_size] = np.max(input_matrix[i:i+pool_size, j:j+pool_size])
    
    return output


def process_and_visualize(x, y, z, directory, sample):
    # Apply a binary threshold on z-values
    thresh = threshold_otsu(z)
    binary_z = z > thresh
    
    # Filter out the bright points
    bright_x, bright_y, bright_z = x[binary_z], y[binary_z], z[binary_z]
    
    # Apply KMeans clustering on bright points' intensities
    kmeans = KMeans(n_clusters=2, random_state=0)
    bright_labels = kmeans.fit_predict(bright_z.reshape(-1, 1))
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=1, ncols=3)
    # Display scatter plot with z-values as color (before thresholding)
    scatter = axes[0].scatter(x, y, c=z, s=5, cmap='inferno')
    axes[0].set_title("Intensity Visualization")
    axes[0].axis('equal')
    plt.colorbar(scatter, ax=axes[0], label='Intensity')
    
    # Display scatter plot with binary_z-values as color (after thresholding)
    axes[1].scatter(x, y, c=binary_z, s=5, cmap='gray')
    axes[1].set_title("Binary Intensity after Thresholding")
    axes[1].axis('equal')
    
    # Display scatter plot of bright points clustered into two intensity levels
    scatter_2 = axes[2].scatter(bright_x, bright_y, c=bright_labels, s=5, cmap='viridis')
    axes[2].set_title("Two Bright Intensity Clusters")
    axes[2].axis('equal')
    plt.colorbar(scatter_2, ax=axes[2], ticks=[0, 1], label='Intensity Cluster')
    
    # Save the figure
    save_path = os.path.join(directory, f'combined_{sample}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.show()
    
    return binary_z


def sample_parameters(shape, mean_range=(0.5, 1.0), std_range=(0.05, 0.1), force_negative_off_diagonal=False):
    """
    Generate random mean and standard deviation values for a given shape.
    
    Parameters:
    - shape: Shape of the matrix or array.
    - mean_range: Tuple indicating the range of mean values.
    - std_range: Tuple indicating the range of standard deviation values.
    - force_negative_off_diagonal: If True, forces the off-diagonal elements of a 2x2 matrix to be negative.
    
    Returns:
    - Random mean and standard deviation matrices/arrays of the given shape.
    """
    mean = np.random.uniform(mean_range[0], mean_range[1], shape)
    std = np.random.uniform(std_range[0], std_range[1], shape)
    # Generating ratio1 and ratio2 as random integers between 5 and 10
    ratio1 = np.random.randint(5, 11) # randint's upper limit is exclusive
    ratio2 = np.random.randint(5, 11)
    
    if shape == (2, 2):
       mean[0, 1] /= ratio1
       mean[1, 0] /= ratio2
       if force_negative_off_diagonal:
        mean[0, 1] = -abs(mean[0, 1])
        mean[1, 0] = -abs(mean[1, 0])

    return mean, std


def sample_from_normal(mean, std, shape):
    return np.random.normal(mean, std, shape)




def transition_new(st, res, signal):
    """
    Transforms array of electron configurations from energy_tensor into a stability diagram with added noise
    :param st: array of electron configuration (output from energy_tensor)
    :param res: resolution (number of pixels)
    :param signal: average signal intensity, defines signal to noise ratio
    :param blur: number of pixels to blur the sample by
    :return: intensity of stability diagram
    """
    transition_intensity = 250

    # Create blank images
    i1, i2 = np.zeros(shape=(res, res)), np.zeros(shape=(res, res))
    
    # Detect vertical transitions and set them in i1
    x1, y1 = np.where(st[:-1] != st[1:])
    i1[x1-1, y1-1] = transition_intensity 
    
    # Detect horizontal transitions and set them in i2
    x2, y2 = np.where(np.transpose(st)[:-1] != np.transpose(st)[1:])
    i2[x2-1, y2-1] = transition_intensity
    
    # Combine i1 and i2
    combined_image = i1 + np.transpose(i2)
    
    # Display the combined image
    plt.imshow(combined_image, cmap='gray')
    plt.show()
    
    return combined_image



def stability_diagram_new(freq,c, cc, n, v, e1, dots, offset):
    """
     Generates stability diagram given capacitance matrix
     @param offset: voltage offset that might be applied
     @param dots: QDs being probed
     @param freq: number of repeating honeycombs in stability diagram
     @param v: voltages being applied
     @param n: electron configurations being taken into consideration
     @param cc: cross capacitance matrix
     @param c: capacitance matrix
     @return: stability diagram
     """
    signal, blur = np.random.uniform(50, 100, 1), 5
    st = energy_tensor(n, v, c, cc)
    intensity = transition_new(st, (len(st) - 1), signal)
    x, y, z = matrix_to_array(intensity)
    x = x / cc[dots[0], dots[0]] / (len(st) - 1) * freq + int(offset[0]) / cc[dots[0], dots[0]]
    y = y / cc[dots[1], dots[1]] / (len(st) - 1) * freq + int(offset[1]) / cc[dots[1], dots[1]]
    return x, y, z,intensity



def generate_and_plot(freq, mean_c, std_c, mean_cg, std_cg, mean_ccs, std_ccs,sample_num):
    matrices = []
    # Sample values from normal distributions
    c = sample_from_normal(mean_c, std_c, (2, 2))
    #matrix x has to be symmetric
    c[0, 1] = -abs(c[0, 1])
    c[1, 0] = c[0, 1]
    cg = sample_from_normal(mean_cg, std_cg, 2)
    ccs = sample_from_normal(mean_ccs, std_ccs, (2, 2))
    ccs=abs(ccs) #make sure cross capacitance is always positive
    ccs[0,0],ccs[1,1] = cg[0],cg[1]
    n = n_states(2, freq, freq)
    res = 400
    v = voltage(2, freq, res, n, cg, [0, 1])
    offset = np.random.uniform(-1, 1, 5)  # Randomly generate offset
    # Recompute the stability diagram
    x, y, z,intensity = stability_diagram_new(freq, c, ccs, n, v, freq, [0, 1], offset)
    # Create the figure and axis
    #Generate highlighted matrix based on intensity
    highlighted_matrix = detect_intersection(intensity, stride=1)
    
    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    
    # Plot for Original Matrix
    cax1 = axes[0].matshow(intensity, cmap='viridis')
    axes[0].set_title("Original Matrix")
    plt.colorbar(cax1, ax=axes[0])
    
    # Plot for Highlighted Matrix
    cax2 = axes[1].matshow(highlighted_matrix, cmap='viridis')
    axes[1].set_title("Highlighted Matrix")
    plt.colorbar(cax2, ax=axes[1])
    
    # Save the figure
    filename = f'sample_{sample_num+1}.png'
    directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data_figs"
    plt.tight_layout()
    plt.savefig(os.path.join(directory, filename), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    matrices.append((x, y, z, c, ccs, n, v,offset))
        
    return matrices,intensity

def hough_transform_and_save(num_samples):
    start_time = datetime.datetime.now()
    intensity_all=[]
    max_input_all=[]
    for sample_num in range(num_samples): 
        mean_c, std_c = sample_parameters((2, 2),force_negative_off_diagonal=True)
        mean_cg, std_cg = sample_parameters(2)
        mean_ccs, std_ccs = sample_parameters((2, 2))
        freq = np.random.randint(3,6)
        # Generate the stability diagram and retrieve matrices and offset
        matrices,intensity= generate_and_plot(freq, mean_c, std_c, mean_cg, std_cg, mean_ccs, std_ccs,sample_num)
        x, y, z, c, ccs, n, v,offset = matrices[0]
        # Compute the Hough transform of the stability diagram
        directory=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data"
        #binary_z=process_and_visualize(x, y, z, directory,sample_num)
        image_side = int(np.sqrt(z.size))
        x_2d = x.reshape(image_side, image_side)
        y_2d = y.reshape(image_side, image_side)
        z_2d = z.reshape(image_side, image_side)
        h, theta, d = hough_line(z_2d)
        # Extract peaks from Hough transform
        _, angles, dists = hough_line_peaks(h, theta, d)
        # Convert the Hough data to a dataframe
        hough_data = [{'angle': angle, 'distance': dist} for angle, dist in zip(angles, dists)]
        df_hough = pd.DataFrame(hough_data)
        hough_dict = df_hough.to_dict(orient='list')

        data = {
            'c': c.tolist(),
            'ccs': ccs.tolist(),
            'freq': freq,
            'offset': offset.tolist()
        }

       # Combine data and hough_dict
        combined_data = {
            'data': data,
            'hough_data': hough_dict
        }
        # Save the combined_data to a pickle file
        
        filename_pickle = os.path.join(directory, f'sample_{sample_num+1}.pkl')
        with open(filename_pickle, 'wb') as file:
            pickle.dump(combined_data, file)
           
        print(f'<<sample : {sample_num+1} saved!>>')
       
        input_matrix=intensity
        highlighted_matrix = detect_intersection(input_matrix,stride=1)
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
        cax1 = axes[0].matshow(input_matrix, cmap='viridis')
        axes[0].set_title("Original Matrix")
        plt.colorbar(cax1, ax=axes[0])
        cax2 = axes[1].matshow(highlighted_matrix, cmap='viridis')
        axes[1].set_title("After Padded Max Pooling")
        plt.colorbar(cax2, ax=axes[1])
        max_intensity=padded_max_pooling(intensity,pool_size=3, stride=1)
        cax3 = axes[2].matshow(max_intensity, cmap='viridis')
        axes[2].set_title("After Padded Max Pooling")
        plt.colorbar(cax3, ax=axes[2])
        max_input=process_submatrices(intensity, stride=1)
        cax4 = axes[3].matshow(max_input, cmap='viridis')
        axes[3].set_title("After Padded Max Pooling")
        plt.colorbar(cax4, ax=axes[3])
        plt.tight_layout()
        # highlighted_top_values = highlight_top_n(max_input, n=2)
        # cax5 = axes[4].matshow(highlighted_top_values, cmap='viridis')
        # axes[4].set_title("Highlighted Top Values")
        # plt.colorbar(cax5, ax=axes[4])
        plt.show()
        
        intensity_all.append(input_matrix)    
        max_input_all.append(max_input)    
        
    # Calculate elapsed time
    elapsed_time = datetime.datetime.now() - start_time
    print(f"Time passed: {elapsed_time}")
    return matrices,x_2d,y_2d,z_2d,intensity_all,max_input_all


matrices,x_2d,y_2d,z_2d,intensity_all,max_input_all=hough_transform_and_save(num_samples=2)

# Example usage:
#plot_histogram(matrix, title="Histogram of Intensities", bins=50, highlight_value=None):

# df=[]    
# for i in range(5):
#     values, counts = get_top_values(max_input_all[i])
#     df.append(display_in_table(values, counts))



# bins=10
# highlight_value=None
# matrix=max_input_all[4]
# flattened_values = matrix.flatten()
# title="Histogram of Intensities"
# # Plot the histogram and get the bar objects
# plt.figure(figsize=(10,5))
# n, bin_edges, patches = plt.hist(flattened_values, bins=bins, color='blue', alpha=0.7)

# # Highlight the bars with specific pixel intensities, if required
# if highlight_value is not None:
#     for i in range(len(bin_edges) - 1):
#         if bin_edges[i] <= highlight_value < bin_edges[i + 1]:
#             patches[i].set_facecolor('red')

# # Label the counts above the bars
# for i in range(len(patches)):
#     height = patches[i].get_height()
#     plt.text(patches[i].get_x() + patches[i].get_width() / 2., height + 0.5,
#              '%d' % int(height), ha='center', va='bottom')

# plt.title(title)
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.show()    



# image_path=r'C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data_figs\sample_24.png'
# image = cv2.imread(image_path, 0)
# image =intensity[1]
# # Apply Canny edge detector

# # Normalize the image to be between 0 and 255
# normalized_image = np.interp(image, (image.min(), image.max()), (0, 255))

# # Convert the normalized image to uint8
# uint8_image = normalized_image.astype(np.uint8)

# # Now you can apply the Canny edge detection
# edges = cv2.Canny(uint8_image, 50, 150, apertureSize=3)

# # Detect lines using Hough Line Transform
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
