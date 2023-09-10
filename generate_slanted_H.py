# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:20:25 2023

@author: ehsan
"""

import cv2
import numpy as np
import random
import os
import pickle
import pandas as pd

def rotate_point(point, center, angle):
    x, y = point
    cx, cy = center
    x_new = cx + (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    y_new = cy + (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    return int(x_new), int(y_new)



def draw_slanted_rotated_H(img, pt1, pt2, trapezoid_mode=True):
   
   right_pt, left_pt = (pt2, pt1) if pt2[0] > pt1[0] else (pt1, pt2)
   
   angle = random.uniform(0, np.pi/2)  # acute angle for trapezoid creation
   
   # Calculate Right Point coordinates
   distance = random.randint(10, 50)
   top_right_pt = (int(right_pt[0] + distance * np.cos(angle)), int(right_pt[1] - distance * np.sin(angle)))
   bottom_right_pt = (int(right_pt[0] + distance * np.cos(-angle)), int(right_pt[1] - distance * np.sin(-angle)))

   # Calculate Left Point coordinates
   top_left_pt = (int(left_pt[0] + distance * np.cos(np.pi - angle)), int(left_pt[1] - distance * np.sin(np.pi - angle)))
   bottom_left_pt = (int(left_pt[0] + distance * np.cos(-np.pi + angle)), int(left_pt[1] - distance * np.sin(-np.pi + angle)))
   
   # Decide on the rotation center (let's choose the center of the image for simplicity)
   center = (img.shape[1]//2, img.shape[0]//2)
   rotation_angle = random.uniform(0, np.pi/2)
   
   # Rotate every point
   pt1_rotated = rotate_point(pt1, center, rotation_angle)
   pt2_rotated = rotate_point(pt2, center, rotation_angle)
   top_right_pt_rotated = rotate_point(top_right_pt, center, rotation_angle)
   bottom_right_pt_rotated = rotate_point(bottom_right_pt, center, rotation_angle)
   top_left_pt_rotated = rotate_point(top_left_pt, center, rotation_angle)
   bottom_left_pt_rotated = rotate_point(bottom_left_pt, center, rotation_angle)

   # Draw the rotated trapezoid
   cv2.line(img, pt1_rotated, pt2_rotated, (255,255,255), 2)
   cv2.line(img, pt1_rotated, top_left_pt_rotated, (255,255,255), 2)
   cv2.line(img, pt1_rotated, bottom_left_pt_rotated, (255,255,255), 2)
   cv2.line(img, pt2_rotated, top_right_pt_rotated, (255,255,255), 2)
   cv2.line(img, pt2_rotated, bottom_right_pt_rotated, (255,255,255), 2)


   return img


def draw_slanted_H(img, pt1, pt2, trapezoid_mode=True):
    
    cv2.line(img, pt1, pt2, (255,255,255), 2)
    
    right_pt, left_pt = (pt2, pt1) if pt2[0] > pt1[0] else (pt1, pt2)
    
    if trapezoid_mode:
        angle = random.uniform(0, np.pi/2)  # acute angle
        
        # Right Point
        distance = random.randint(10, 50)
        top_right_pt = (int(right_pt[0] + distance * np.cos(angle)), int(right_pt[1] - distance * np.sin(angle)))
        cv2.line(img, right_pt, top_right_pt, (255,255,255), 2)
        
        bottom_right_pt = (int(right_pt[0] + distance * np.cos(-angle)), int(right_pt[1] - distance * np.sin(-angle)))
        cv2.line(img, right_pt, bottom_right_pt, (255,255,255), 2)

        # Left Point
        top_left_pt = (int(left_pt[0] + distance * np.cos(np.pi - angle)), int(left_pt[1] - distance * np.sin(np.pi - angle)))
        cv2.line(img, left_pt, top_left_pt, (255,255,255), 2)

        bottom_left_pt = (int(left_pt[0] + distance * np.cos(-np.pi + angle)), int(left_pt[1] - distance * np.sin(-np.pi + angle)))
        cv2.line(img, left_pt, bottom_left_pt, (255,255,255), 2)

    else:
        
        right_pt, left_pt = (pt2, pt1) if pt2[0] > pt1[0] else (pt1, pt2)
        angle = random.uniform(0, np.pi/2)
        distance = random.randint(10, 50)  # random distance
        end_pt = (int(right_pt[0] + distance * np.cos(angle)), int(right_pt[1] - distance * np.sin(angle)))
        cv2.line(img, right_pt, end_pt, (255,255,255), 2)

        angle = random.uniform(-np.pi/2, 0)
        distance = random.randint(10, 50)  # random distance
        end_pt = (int(right_pt[0] + distance * np.cos(angle)), int(right_pt[1] - distance * np.sin(angle)))
        cv2.line(img, right_pt, end_pt, (255,255,255), 2)

        #For leftmost point
        angle = random.uniform(np.pi/2, np.pi)
        distance = random.randint(10, 50)  # random distance
        end_pt = (int(left_pt[0] + distance * np.cos(angle)), int(left_pt[1] - distance * np.sin(angle)))
        cv2.line(img, left_pt, end_pt, (255,255,255), 2)

        angle = random.uniform(np.pi, 1.5*np.pi)
        distance = random.randint(10, 50)  # random distance
        end_pt = (int(left_pt[0] + distance * np.cos(angle)), int(left_pt[1] - distance * np.sin(angle)))
        cv2.line(img, left_pt, end_pt, (255,255,255), 2)

    
    #Rotate the figure
    #Get the rotation center (here, it's the center of the image, but you can change it if needed)
    center = ((right_pt[0]+left_pt[0])//2, (right_pt[1]+left_pt[1])//2)
    
    # Generate a random rotation angle between 0 and pi/2 (converted to degrees)
    rotation_angle = random.uniform(0, np.pi/2) * 180.0 / np.pi  # Convert to degrees
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    
    # Apply the rotation
    img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    return img


def generate_samples(num_samples, directory_data="output", directory_image="output2", img_size=(512, 512), equal_y=True):
    data_list = []
    for i in range(num_samples):
        img = np.zeros(img_size, dtype=np.uint8)
        
        if equal_y:
            y = random.randint(0, img_size[1]-1)
            pt1 = (random.randint(0, img_size[0]-1), y)
            pt2 = (random.randint(0, img_size[0]-1), y)
        else:
            pt1 = (random.randint(0, img_size[0]-1), random.randint(0, img_size[1]-1))
            pt2 = (random.randint(0, img_size[0]-1), random.randint(0, img_size[1]-1))
        
        img=draw_slanted_H(img, pt1, pt2)
        #draw_slanted_rotated_H(img, pt1, pt2)
        # Save each image directly within this loop
        if not os.path.exists(directory_image):
            os.makedirs(directory_image)
        image_filename = os.path.join(directory_image, f"sample_{i}.png")
        cv2.imwrite(image_filename, img)
        
        # Append data for pickle saving
        # data.append({
        #     'image': image_filename,  # Storing the path instead of the image data to reduce pickle size
        #     'triple_points': (pt1, pt2)
        # })
        
        data_list.append([ img, np.array([pt1[0], pt1[1], pt2[0], pt2[1]])])
        print(f"Saved image to: {image_filename}")

     # Create a DataFrame from the data
    df = pd.DataFrame(data_list, columns=['image_filename', 'coordinates'])

    # Save the DataFrame as a pickle file
    if not os.path.exists(directory_data):
        os.makedirs(directory_data)
    pickle_filename = os.path.join(directory_data, "data.pkl")
    with open(pickle_filename, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved data to: {pickle_filename}")

    return df



directory_data=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\slanted_H_training_data"
directory_image=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\slanted_H_training_figs"
df = generate_samples(2000, directory_data=directory_data, directory_image=directory_image,img_size=(400,400), equal_y=True)
      

