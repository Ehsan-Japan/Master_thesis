# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:17:06 2023

@author: ehsan
"""

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

import random
from preprocess import *


def plot_stab(x, y, volt, dots, **kwargs):
    z = kwargs.get('z', None)
    val = dots + np.ones(2)
    if z is not None:
        plt.scatter(x, y, c=z, s=5, cmap='inferno')
    else:
        plt.scatter(x, y, c='k', s=5)
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.gca().set_aspect('equal', adjustable='box')
    if volt == 'V':
        plt.xlabel(r'$V_{g%s}$ (V)' % int(val[0]), fontsize=24)
        plt.ylabel(r'$V_{g%s}$ (V)' % int(val[1]), fontsize=24)
    elif volt == 'U':
        plt.xlabel(r'$U_{%s}$ (V)' % int(val[0]), fontsize=24)
        plt.ylabel(r'$U_{%s}$ (V)' % int(val[1]), fontsize=24)
    #plt.tight_layout()

   
def sweep_and_plot(interval_ccs01, interval_ccs10, increment_ccs01, increment_ccs10, directory, c2, cg2, ccs2):
    # Extract interval values for clarity
    start_ccs01, end_ccs01 = interval_ccs01
    start_ccs10, end_ccs10 = interval_ccs10
    
    # Create arrays of values to sweep over
    values_ccs01 = np.arange(start_ccs01, end_ccs01 + increment_ccs01, increment_ccs01)
    values_ccs10 = np.arange(start_ccs10, end_ccs10 + increment_ccs10, increment_ccs10)
    
    for val_ccs01 in values_ccs01:
        for val_ccs10 in values_ccs10:
            # Update ccs matrix values
            ccs2[0,1] = val_ccs01
            ccs2[1,0] = val_ccs10
            
            # Recompute the stability diagram with the new ccs matrix
            freq = 6
            n2 = n_states(2, freq, freq )
            v2 = voltage(2, freq, res, n2, cg2, [0, 1])
            x2, y2, z2 = stability_diagram(c2, ccs2, n2, v2, freq, [0, 1], offset)
            
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(20, 15))
            
            # Add the stability plot
            plot_stab(x2, y2, 'V', [0, 1], z=z2, ax=ax)
            ax.set_title(f'ccs[0,1] = {val_ccs01:.2f}, ccs[1,0] = {val_ccs10:.2f}')
            
            # Create tables with the matrices
            table_c2_data = [["", "0", "1"], ["0", c2[0,0], c2[0,1]], ["1", c2[1,0], c2[1,1]]]
            table_cg2_data = [["Index", "Value"], ["0", cg2[0]], ["1", cg2[1]]]
            table_ccs2_data = [["", "0", "1"], ["0", ccs2[0,0], ccs2[0,1]], ["1", ccs2[1,0], ccs2[1,1]]]
            
            ax_table_c2 = fig.add_axes([0.01, 0.7, 0.2, 0.2])  # Add table for c2
            ax_table_c2.axis('off')
            ax_table_c2.table(cellText=table_c2_data, cellLoc='center', loc='center', colWidths=[0.1, 0.1, 0.1])
            ax_table_c2.set_title('c2 Matrix')
            
            ax_table_cg2 = fig.add_axes([0.01, 0.4, 0.2, 0.2])  # Add table for cg2
            ax_table_cg2.axis('off')
            ax_table_cg2.table(cellText=table_cg2_data, cellLoc='center', loc='center', colWidths=[0.1, 0.1])
            ax_table_cg2.set_title('cg2 Array')
            
            ax_table_ccs2 = fig.add_axes([0.01, 0.1, 0.2, 0.2])  # Add table for ccs2
            ax_table_ccs2.axis('off')
            ax_table_ccs2.table(cellText=table_ccs2_data, cellLoc='center', loc='center', colWidths=[0.1, 0.1, 0.1])
            ax_table_ccs2.set_title('ccs2 Matrix')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(directory, f'[{val_ccs01:.2f},{val_ccs10:.2f}].png'))
            plt.close()  # Close the figure to free up memory


#sweep_and_plot((0,2), (0,2), 0.2, 0.2)

def generate_sweep_values(start, end, increment):
    return np.arange(start, end + increment, increment)

import os

def sweep_parameters_and_save(
    interval_c2_00, interval_c2_11, interval_c2_01, 
    interval_cg2_0, interval_cg2_1,
    interval_ccs2_00, interval_ccs2_11, interval_ccs2_01,
    increment
):
    values_c2_00 = generate_sweep_values(*interval_c2_00, increment)
    values_c2_11 = generate_sweep_values(*interval_c2_11, increment)
    values_c2_01 = generate_sweep_values(*interval_c2_01, increment)  # also c2_10 due to symmetry
    
    values_cg2_0 = generate_sweep_values(*interval_cg2_0, increment)
    values_cg2_1 = generate_sweep_values(*interval_cg2_1, increment)
    
    values_ccs2_00 = generate_sweep_values(*interval_ccs2_00, increment)
    values_ccs2_11 = generate_sweep_values(*interval_ccs2_11, increment)
    values_ccs2_01 = generate_sweep_values(*interval_ccs2_01, increment)  # also ccs2_10 due to symmetry
    
    base_directory = r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\Figs_new"
    
    for val_c2_00 in values_c2_00:
        for val_c2_11 in values_c2_11:
            for val_c2_01 in values_c2_01:
                for val_cg2_0 in values_cg2_0:
                    for val_cg2_1 in values_cg2_1:
                        for val_ccs2_00 in values_ccs2_00:
                            for val_ccs2_11 in values_ccs2_11:
                                for val_ccs2_01 in values_ccs2_01:
                                    
                                    # Update matrices/values
                                    c2 = np.array([[val_c2_00, val_c2_01], [val_c2_01, val_c2_11]])
                                    cg2 = np.array([val_cg2_0, val_cg2_1])
                                    ccs2 = np.array([[val_ccs2_00, val_ccs2_01], [val_ccs2_01, val_ccs2_11]])
                                    
                                    # Create folder for this combination
                                    folder_name = f"c2_{val_c2_00:.2f}_{val_c2_11:.2f}_{val_c2_01:.2f}_cg2_{val_cg2_0:.2f}_{val_cg2_1:.2f}_ccs2_{val_ccs2_00:.2f}_{val_ccs2_11:.2f}_{val_ccs2_01:.2f}"
                                    current_directory = os.path.join(base_directory, folder_name)
                                    if not os.path.exists(current_directory):
                                        os.makedirs(current_directory)
                                    
                                    # Call sweep_and_plot function to generate figures
                                    sweep_and_plot((interval_ccs2_01[0], interval_ccs2_01[1]), (interval_ccs2_01[0], interval_ccs2_01[1]), increment, increment, current_directory, c2, cg2, ccs2)





# Define intervals and increment for the sweeps

# For c2 matrix
interval_c2_00 = (0.1, 1.0)  # Start and end values for c2[0,0]
interval_c2_11 = (0.1, 1.0)  # Start and end values for c2[1,1]
interval_c2_01 = (0.1, 0.5)  # Start and end values for c2[0,1] (and c2[1,0] due to symmetry)

# For cg2 array
interval_cg2_0 = (0.1, 0.5)  # Start and end values for cg2[0]
interval_cg2_1 = (0.1, 0.5)  # Start and end values for cg2[1]

# For ccs2 matrix
interval_ccs2_00 = (0.1, 0.3)  # Start and end values for ccs2[0,0]
interval_ccs2_11 = (0.1, 0.3)  # Start and end values for ccs2[1,1]
interval_ccs2_01 = (0.1, 0.3)  # Start and end values for ccs2[0,1] (and ccs2[1,0] due to symmetry)

# Increment for the sweep (assuming a common increment for simplicity)
increment = 0.2

# Call the function
sweep_parameters_and_save(
    interval_c2_00, interval_c2_11, interval_c2_01, 
    interval_cg2_0, interval_cg2_1,
    interval_ccs2_00, interval_ccs2_11, interval_ccs2_01,
    increment
)
