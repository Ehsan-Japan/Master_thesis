# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:08:39 2023

@author: ehsan
"""
import os
import pickle
import pandas as pd
import numpy as np

directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\features"
filename = "sample_1000.pkl"
full_path = os.path.join(directory, filename)

with open(full_path, 'rb') as file:
    data = pickle.load(file)

directory_labels=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data"


def unpack_all_pickles_to_dataframe(directory):
    all_rows = []

    i = 1
    while True:
        file_name = os.path.join(directory, f'sample_{i}.pkl')
        
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                combined_data = pickle.load(f)
                data = combined_data['data']
                
                # Flatten the matrices using numpy
                c_flattened = np.array(data['c']).flatten().tolist()
                ccs_flattened = np.array(data['ccs']).flatten().tolist()
                
                freq = [data['freq']]
                #offset = data['offset']

                # Concatenate all lists
                row = c_flattened + ccs_flattened + freq

                # Append to the all_rows list
                all_rows.append(row)

            i += 1
        else:
            break

    # Convert list of rows to a DataFrame
    df = pd.DataFrame(all_rows)

    return df,data['c'],data['ccs']

# Example usage

#df,c,ccs= unpack_all_pickles_to_dataframe(directory_labels)


all_rows = []

i = 1
directory=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\training_data"
file_name = os.path.join(directory, f'sample_{i}.pkl')
with open(file_name, 'rb') as f:
    combined_data = pickle.load(f)
    data = combined_data['data']
    
    # Flatten the matrices using numpy
    c_flattened = np.array(data['c']).flatten().tolist()
    ccs_flattened = np.array(data['ccs']).flatten().tolist()
    
    freq = [data['freq']]
    #offset = data['offset']

    # Concatenate all lists
    row = c_flattened + ccs_flattened + freq

    # Append to the all_rows list
    all_rows.append(row)

