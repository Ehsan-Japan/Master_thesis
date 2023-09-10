# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:18:03 2023

@author: ehsan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:16:09 2023

@author: ehsan
"""
import numpy as np
import stability as stab
import matplotlib.pyplot as plt
import itertools
from scipy.signal import convolve
import os
import numpy as np
import cv2
import sys
import pandas as pd



def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y




# # Values of the capacitance used within the paper:
# C = np.array([[ 1.61985818, -0.40839002, -0.06621002, -0.0363675 ],
#               [-0.40839002,  1.85134262, -0.05580086, -0.30766702],
#               [-0.06621002, -0.05580086,  1.68453668, -0.38063206],
#               [-0.0363675 , -0.30766702, -0.38063206,  1.87725365]])

# CC = np.array([[1.02248092, 0.04857599, 0.02723455, 0.01059919],
#                [0.05873536, 0.95193755, 0.0118957 , 0.05691611],
#                [0.04809363, 0.03219296, 1.05491902, 0.04668813],
#                [0.04828182, 0.02871698, 0.09730076, 0.97828752]])
# Cg = np.array([CC[0,0], CC[1,1], CC[2,2], CC[3,3]])


'''
To obtain a random set of capacitor values, uncoment the following piece of code,
although the offset voltages and electron numbers might need tweeking accordingly
'''
# c = sorted(np.random.uniform(1, 0.01, 3))
# C, Cg, CC, CM = stab.random_c(c[2], c[0], c[1], N_QD, sorted(np.random.uniform(0,1,3)))

'''
Since we are applying higer voltages, we are going to add electrons to the dots being probed accordingly 
this is to reduce the amount of RAM required to run the algorithem.
This can be tuned by accordingly by chechink the electron configuration within st
'''

def rand_c(mu_c,sigma_c, seed):
    if seed is not None:
        np.random.seed(seed)
    return abs(np.random.normal(mu_c, sigma_c, 1))




def generate_random_uniform(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1, 2)

def generate_random_normal(mean, std_dev, size, seed):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std_dev, size)


def random_c(ci, cc, cm, n_qd, ratio, seed):
    """
    Generates random capacitance matrix when inputting an average value for:
    @param ci: Gate capacitance
    @param cc: Cross capacitance
    @param cm: Mutual capacitance
    @param n_qd: number of QDs
    @return: c, capacitance matrix
    @param ratio: ratio between capacitance parallel, perpendicular and diagonally across the nano-wire
    @param seed: random seed for reproducibility
    """     
    # Setting up capacitance and cross capacitance matrices
    c = np.zeros(shape=(n_qd, n_qd))
    cg = abs(generate_random_normal(ci, ci / 5, int(n_qd), seed))  # Gate capacitance using the seeded random generator
    ccs = np.identity(n_qd) * cg
    # capacitance perpendicular to the nano-wire
    for i in range(int(n_qd // 2)):
        c[i * 2, i * 2 + 1] = c[i * 2 + 1, i * 2] = -rand_c(cm, ratio[2], seed)
        ccs[i * 2, i * 2 + 1], ccs[i * 2 + 1, i * 2] = rand_c(cc, ratio[2], seed), rand_c(cc, ratio[2], seed)
    # capacitance parallel to the nano-wire
    for j in range(n_qd - 2):
        c[j, j + 2] = c[j + 2, j] = -rand_c(cm, ratio[1], seed)
        ccs[j, j + 2], ccs[j + 2, j] = rand_c(cc, ratio[1], seed), rand_c(cc, ratio[1], seed)
    # capacitance diagonally across the nano-wire
    for k in range((n_qd - 2) // 2):
        c[2 * k, 2 * k + 3] = c[2 * k + 3, 2 * k] = -rand_c(cm, ratio[0], seed)
        c[2 * k + 1, 2 * k + 2] = c[2 * k + 2, 2 * k + 1] = -rand_c(cm, ratio[0], seed)
        c[k, 3 - k] = c[3 - k, k] = -rand_c(cm, ratio[0], seed)
        ccs[2 * k, 2 * k + 3], ccs[2 * k + 3, 2 * k] = rand_c(cc, ratio[0], seed), rand_c(cc, ratio[0], seed)
        ccs[2 * k + 1, 2 * k + 2], ccs[2 * k + 2, 2 * k + 1] = rand_c(cc, ratio[0], seed), rand_c(cc, ratio[0], seed)
    # Total capacitance on dot i
    for i in range(0, n_qd):
        c[i, i] = np.sum(abs(c[i])) + np.sum(ccs[i])
        
        
    if np.linalg.det(c) != 0 and np.linalg.det(ccs) != 0:
            return c, cg, ccs    
    #return c, cg, ccs

def capacitance(ci, cc, cm,cg, n_qd, ratio,seed):
    """
    Generates random capacitance matrix when inputting an average value for:
    @param ci: Gate capacitance
    @param cc: Cross capacitance
    @param cm: Mutual capacitance
    @param n_qd: number of QDs
    @return: c, capacitance matrix
    @param ratio: ratio between capacitance parallel, perpendicular and diagonally across the nano-wire
    @param seed: random seed for reproducibility
    """     
    # Setting up capacitance and cross capacitance matrices
    c = np.zeros(shape=(n_qd, n_qd))
    #cg = abs(generate_random_normal(ci, ci / 5, int(n_qd), seed))  # Gate capacitance using the seeded random generator
    ccs = np.identity(n_qd) * cg
    # capacitance perpendicular to the nano-wire
    for i in range(int(n_qd // 2)):
        c[i * 2, i * 2 + 1] = c[i * 2 + 1, i * 2] = -rand_c(cm, ratio[2], seed)
        ccs[i * 2, i * 2 + 1], ccs[i * 2 + 1, i * 2] = rand_c(cc, ratio[2], seed), rand_c(cc, ratio[2], seed)
    # capacitance parallel to the nano-wire
    for j in range(n_qd - 2):
        c[j, j + 2] = c[j + 2, j] = -rand_c(cm, ratio[1], seed)
        ccs[j, j + 2], ccs[j + 2, j] = rand_c(cc, ratio[1], seed), rand_c(cc, ratio[1], seed)
    # capacitance diagonally across the nano-wire
    for k in range((n_qd - 2) // 2):
        c[2 * k, 2 * k + 3] = c[2 * k + 3, 2 * k] = -rand_c(cm, ratio[0], seed)
        c[2 * k + 1, 2 * k + 2] = c[2 * k + 2, 2 * k + 1] = -rand_c(cm, ratio[0], seed)
        c[k, 3 - k] = c[3 - k, k] = -rand_c(cm, ratio[0], seed)
        ccs[2 * k, 2 * k + 3], ccs[2 * k + 3, 2 * k] = rand_c(cc, ratio[0], seed), rand_c(cc, ratio[0], seed)
        ccs[2 * k + 1, 2 * k + 2], ccs[2 * k + 2, 2 * k + 1] = rand_c(cc, ratio[0], seed), rand_c(cc, ratio[0], seed)
    # Total capacitance on dot i
    for i in range(0, n_qd):
        c[i, i] = np.sum(abs(c[i])) + np.sum(ccs[i])
        
        
    if np.linalg.det(c) != 0 and np.linalg.det(ccs) != 0:
            return c, cg, ccs   














def random_c_new(c_d0,c_d2,c_d0d2,c_d0v3,c_d2v1):
    """
    Generates random capacitance matrix when inputting an average value for:
    @param ci: Gate capacitance
    @param cc: Cross capacitance
    @param cm: Mutual capacitance
    @param n_qd: number of QDs
    @return: c, capacitance matrix
    @param ratio: ratio between capacitance parallel, perpendicular and diagonally across the nano-wire
    @param seed: random seed for reproducibility
    """     
    # Setting up capacitance and cross capacitance matrices
    epsilon = np.finfo(float).eps
    
    ci=0.3
    cg = abs(generate_random_normal(ci, ci / 5, 4,seed=0))
    
    c,ccs = np.zeros(shape=(4, 4)),np.zeros(shape=(4, 4))    
    c[0,0],c[2,2]=c_d0,c_d2
    c[0,2]=c[2,0]=c_d0d2

    
    #Cross capacitance
    ccs[0,3]=ccs[3,0]=c_d0v3
    ccs[1,2]=ccs[2,1]=c_d2v1
    
    #Gate Capacitance
    ccs[0,0],ccs[2,2]=cg[0],cg[2]
    
    # Replace zeros with epsilon
    c[c == 0] = epsilon
    ccs[ccs == 0] = epsilon
    
    return c, cg, ccs

def random_c_new_non_singular(c_d0, c_d2, c_d0d2, c_d0v3, c_d2v1):
    epsilon = np.finfo(float).eps
    
    ci = 0.3
    cg = abs(np.random.normal(ci, ci / 5, 4))
    
    while True:
        c, ccs = np.zeros(shape=(4, 4)), np.zeros(shape=(4, 4))
        
        c[0, 0], c[2, 2] = c_d0, c_d2
        c[0, 2] = c[2, 0] = c_d0d2
        
        ccs[0, 3] = ccs[3, 0] = c_d0v3
        ccs[1, 2] = ccs[2, 1] = c_d2v1
        ccs[0, 0], ccs[2, 2] = cg[0], cg[2]
        
        c[c == 0] = epsilon
        ccs[ccs == 0] = epsilon
        
        if np.linalg.det(c) != 0 and np.linalg.det(ccs) != 0:
            return c, cg, ccs

def n_states(n_qd: int, max_e: int, diff: int):
    """
    Determines all possible electron configurations within a stability map
    :param n_qd: number of QDs
    :param max_e: maximum number of electrons within a specific QD
    :param diff: maximum difference between the two most populated QDs
    :return: n_st, all possible electron configurations considered
    """
    # Number of possible electron configurations taken into account,
    # the range determines the maximum electron configuration taken into account
    n_st = np.fromiter(itertools.product(range(0, max_e), repeat=n_qd), np.dtype('u1,' * n_qd))
    n_st = n_st.view('u1').reshape(-1, n_qd)

    # Eliminates transitions that are unlikely to be observed in stability
    # map range that is calculated. Can change the difference between max1 and max2
    # or completely comment this part out if needed
    n = np.array([])
    for i in range(0, len(n_st)):
        max1 = n_st[i, np.argmax(n_st[i])]
        max2 = np.delete(n_st[i], np.argmax(n_st[i]))
        max2 = max2[np.argmax(max2)]
        if max1 <= max2 + diff:
            n = np.append(n, n_st[i])

    return np.reshape(n, (int(len(n) / n_qd), n_qd))

def reduced_n(n_st, el, dots):
    n = n_states(2, el + 1, el)
    ns = np.zeros((len(n_st) * len(n), 4))
    ns[:, dots[0]] = np.tile(n_st[:, 0], len(n))
    ns[:, dots[1]] = np.tile(n_st[:, 1], len(n))
    nt = []
    for i in range(len(n)):
        nt = np.append(nt, np.ones(np.shape(n_st)) * n[i])
    nt = np.reshape(nt, (len(n_st) * len(n), 2))
    diff = np.setdiff1d([0, 1, 2, 3], dots)
    ns[:, diff[0]], ns[:, diff[1]] = nt[:, 0], nt[:, 1]
    return ns


def voltage(n_qd, freq, res, n, cg, dots):
    """
    Creates numpy array with applied voltages considered
    :param n_qd: number of QDs
    :param freq: number of repeating honeycombs in stability diagram
    :param res: resolution (number of pixels)
    :param n: possible electron configurations
    :param cg: array of gate capacitance
    :param dots: array of which two QDs are being probed
    :return: numpy array of voltages applied
    """
    vs = np.zeros((n_qd, len(n), res + 1, res + 1))
    v = np.tile(np.arange(0, res + 1, 1) * freq / res, (res + 1, 1))
    vs[dots[0]] = np.tile(v / cg[dots[0]], (len(n), 1, 1))
    vs[dots[1]] = np.tile(np.transpose(v) / cg[dots[1]], (len(n), 1, 1))
    return vs


def stability_diagram(freq,c, cc, n, v, e1, dots, offset):
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
    intensity = transition(st, (len(st) - 1), signal, blur)
    x, y, z = matrix_to_array(intensity)
    x = x / cc[dots[0], dots[0]] / (len(st) - 1) * freq + int(offset[0]) / cc[dots[0], dots[0]]
    y = y / cc[dots[1], dots[1]] / (len(st) - 1) * freq + int(offset[1]) / cc[dots[1], dots[1]]
    return x, y, z

def stability_diagram2(c, cc, n, v, e1,e2, dots, offset):
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
    intensity = transition(st, (len(st) - 1), signal, blur)
    x, y, z = matrix_to_array(intensity)
    x = x / cc[dots[0], dots[0]] / (len(st) - 1) * e1 + int(offset[0]) / cc[dots[0], dots[0]]
    y = y / cc[dots[1], dots[1]] / (len(st) - 1) * e2 + int(offset[1]) / cc[dots[1], dots[1]]
    return x, y, z


def matrix_to_array(int_matrix):
    """
    Converts stability diagram matrix into x, y and intensity arrays
    @param int_matrix: stability diagram matrix
    @return: x, y, intensity
    """
    res_x, res_y = len(int_matrix[0]), len(int_matrix)
    x, y = [], []
    x.extend([np.arange(0, res_x, 1) for i in range(res_y)])
    y.extend([[i] * res_y for i in range(res_x)])
    x, y = np.reshape(x, (res_x * res_y,)), np.reshape(y, (res_x * res_y,))
    intensity = np.reshape(int_matrix, (res_x * res_y,))

    return x, y, intensity


def transition(st, res, signal, blur):
    """
    Transforms array of electron configurations from energy_tensor into a stability diagram with added noise
    :param st: array of electron configuration (output from energy_tensor)
    :param res: resolution (number of pixels)
    :param signal: average signal intensity, defines signal to noise ratio
    :param blur: number of pixels to blur the sample by
    :return: intensity of stability diagram
    """
    # Convert states to transitions
    i1, i2 = np.zeros(shape=(res, res)), np.zeros(shape=(res, res))
    x1, y1 = np.where(st[:-1] != st[1:])
    x2, y2 = np.where(np.transpose(st)[:-1] != np.transpose(st)[1:])
    i1[x1 - 1, y1 - 1] = signal * np.random.uniform(5, 10, 1)
    i2[x2 - 1, y2 - 1] = signal * np.random.uniform(5, 10, 1)

    signal = i1 + np.transpose(i2)  # Pure signal

    # Blur pixels by averaging blur nearest neighbours
    kernel = np.ones((blur, blur)) / blur ** 2
    blurred_signal = convolve(signal, kernel, mode='same')

    # Adding noise to signal
    return add_noise(res, blurred_signal, blur)



def add_noise(res, z, blur):
    gauss = np.random.randn(res, res) / res
    # Adding gaussian and poisson noise to the intensity
    z_n = z + np.random.normal(np.mean(z), 1 / blur, np.shape(z)) + np.random.poisson(np.max(z) + 1 / blur, np.shape(z))
    return 2 * z_n + z_n * gauss  # Adding speckle noise

def energy_tensor(n, v, c, cc):
    """
    Finds the electron configuration within the ones set in N that gives the lowest energy and assigns the index of
    the state of N to that particular value
    :param n: all possible electron configurations considered
    :param v: voltages being applied
    :param c: capacitance matrix
    :param cc: cross capacitance matrix
    :return: 2D array of indices of N that gave the lowest energy for that particular set of voltages
    """
    q_all = np.einsum('ij,jklm', cc, v) - np.tensordot(np.transpose(n), np.ones(np.shape(v[0, 0])), axes=0)
    inverse_c = np.linalg.inv(c)
    volt = np.einsum('ij,jklm', inverse_c, q_all)
    u = 0.5 * np.multiply(q_all, volt)
    energy = u.sum(axis=0)
    return np.argmin(energy, axis=0)

def stab_fqd_revised(res,seed):
    """
    @param res: resolution (number of pixels rastered)
    @return: Stability diagram of a 2x2 QD
    """
    con = 3
    ratio, dots = sorted(np.random.uniform(0.3, 1, 3)), [0, int(np.random.randint(1, 4, 1))]
    while con > 1.25:
        
        rand = generate_random_uniform(seed=seed)
        ci, cm, cc = rand_c(1, 1,seed), rand_c(rand[0], 1,seed), rand_c(rand[0] * rand[1], 1,seed)
        c, cg, ccs = random_c(ci, cc, cm, 4, ratio,seed)
        con = np.linalg.cond(c)
    freq, offset = int(np.random.randint(3, 6, 1)), np.random.randint(1, 7, 2) / cg[dots]
    # Try to reduce amount of RAM required to run
    n = n_states(2, freq + 4, freq + 3)
    ns = reduced_n(n, freq, dots)
    ns[:, dots[0]], ns[:, dots[1]] = ns[:, dots[0]] + int(offset[0]), ns[:, dots[1]] + int(offset[1])
    v = voltage(4, freq, res, ns, cg, dots)
    v[dots[0]], v[dots[1]] = v[dots[0]] + int(offset[0]) / cg[dots[0]], v[dots[1]] + int(offset[1]) / cg[dots[1]]
    x, y, z = stability_diagram(c, ccs, ns, v, freq, dots, offset)
    return x, y, z, c, ccs, dots



def stab_dqd_revised(res,freq,seed,add1,add2,offset,seed_ci,seed_cm,seed_cc,seed_c):
    """
    @param res: resolution (number of pixels rastered)
    @return: Stability diagram of a DQD
    """
    #con = 3
    #while con > 1.25:
    rand=generate_random_uniform(seed)
    ci, cm, cc = rand_c(1, 1,seed_ci), rand_c(rand[0], 1,seed_cm), rand_c(rand[0] * rand[1], 1,seed_cc)
    c, cg, ccs = random_c(ci, cc, cm, 2, np.ones(3),seed_c)
    #con = np.linalg.cond(c)
    #freq = int(np.random.randint(3, 6, 1))
    n = n_states(2, freq + add1, freq + add2)
    v = voltage(2, freq, res, n, cg, [0, 1])
    x, y, z = stability_diagram(c, ccs, n, v, freq, [0, 1], offset)
    return x, y, z, c, ccs, [0, 1],n,v



def stab_dqd(res):
    """
    @param res: resolution (number of pixels rastered)
    @return: Stability diagram of a DQD
    """
    con = 3
    while con > 1.25:
        rand = np.random.uniform(1, 0.1, 2)
        ci, cm, cc = rand_c(1, 1), rand_c(rand[0], 1), rand_c(rand[0] * rand[1], 1)
        c, cg, ccs = random_c(ci, cc, cm, 2, np.ones(3))
        con = np.linalg.cond(c)
    freq = int(np.random.randint(3, 6, 1))
    n = n_states(2, freq + 4, freq + 3)
    v = voltage(2, freq, res, n, cg, [0, 1])
    x, y, z = stability_diagram(c, ccs, n, v, freq, [0, 1], np.zeros(2))
    return x, y, z, c, ccs, [0, 1]


def stab_fqd(res):
    """
    @param res: resolution (number of pixels rastered)
    @return: Stability diagram of a 2x2 QD
    """
    con = 3
    ratio, dots = sorted(np.random.uniform(0.3, 1, 3)), [0, int(np.random.randint(1, 4, 1))]
    while con > 1.25:
        rand = np.random.uniform(1, 0.1, 2)
        ci, cm, cc = rand_c(1, 1), rand_c(rand[0], 1), rand_c(rand[0] * rand[1], 1)
        c, cg, ccs = random_c(ci, cc, cm, 4, ratio)
        con = np.linalg.cond(c)
    freq, offset = int(np.random.randint(3, 6, 1)), np.random.randint(1, 7, 2) / cg[dots]
    # Try to reduce amount of RAM required to run
    n = n_states(2, freq + 4, freq + 3)
    ns = reduced_n(n, freq, dots)
    ns[:, dots[0]], ns[:, dots[1]] = ns[:, dots[0]] + int(offset[0]), ns[:, dots[1]] + int(offset[1])
    v = voltage(4, freq, res, ns, cg, dots)
    v[dots[0]], v[dots[1]] = v[dots[0]] + int(offset[0]) / cg[dots[0]], v[dots[1]] + int(offset[1]) / cg[dots[1]]
    x, y, z = stability_diagram(c, ccs, ns, v, freq, dots, offset)
    return x, y, z, c, ccs, dots



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

def plot_stab_modified(x, y, volt, dots, z, ax=None):
    ax.scatter(x, y, c=z, s=5, cmap='inferno')

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_aspect('equal', adjustable='box')
    
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def average_pooling(input_matrix, pool_size, stride):
    H, W = input_matrix.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(0, H - pool_size + 1, stride):
        for j in range(0, W - pool_size + 1, stride):
            output[i // stride, j // stride] = np.mean(input_matrix[i:i+pool_size, j:j+pool_size])
    
    return output

def padded_max_pooling(input_matrix, pool_size, stride):
    H, W = input_matrix.shape
    
    # Padding the matrix with zeros
    padded_matrix = np.pad(input_matrix, ((0, pool_size - H % pool_size), (0, pool_size - W % pool_size)), mode='constant')
    
    H_pad, W_pad = padded_matrix.shape
    H_out = (H_pad - pool_size) // stride + 1
    W_out = (W_pad - pool_size) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(0, H_pad - pool_size + 1, stride):
        for j in range(0, W_pad - pool_size + 1, stride):
            output[i // stride, j // stride] = np.max(padded_matrix[i:i+pool_size, j:j+pool_size])
    
    return output


def get_max_intensity_info(matrix, top_n=1):
    """Return the top_n highest pixel intensities in descending order and their counts."""
    flattened_values = matrix.flatten()
    
    # Get unique intensities and their counts
    unique_intensities, counts = np.unique(flattened_values, return_counts=True)
    
    # Sort the intensities in descending order
    sorted_indices = np.argsort(unique_intensities)[::-1]  # descending order
    
    # Get the top_n intensities and their counts
    top_intensities = unique_intensities[sorted_indices[:top_n]]
    top_counts = counts[sorted_indices[:top_n]]
    
    return list(zip(top_intensities, top_counts))


def sample_parameters(shape, mean_range=(0.5, 1.0), std_range=(0.05, 0.1), 
                      force_negative_off_diagonal=False):
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
    # Generating ratio1 and ratio2 as random integers between 5 and 10
    ratio1 = np.random.randint(5, 11) # randint's upper limit is exclusive
    ratio2 = np.random.randint(5, 11)
    
    mean = np.random.uniform(mean_range[0], mean_range[1], shape)
    std = np.random.uniform(std_range[0], std_range[1], shape)
    
    if shape == (2, 2):
        mean[0, 1] /= ratio1
        mean[1, 0] /= ratio2
        if force_negative_off_diagonal:
            mean[0, 1] = -abs(mean[0, 1])
            mean[1, 0] = -abs(mean[1, 0])

    return mean, std


def plot_histogram(matrix, title="Histogram of Intensities", bins=50, highlight_value=None):
    """Plot a histogram for a given matrix."""
    # Flatten the matrix into a 1D array for histogram plotting
    flattened_values = matrix.flatten()
    
    # Plot the histogram and get the bar objects
    plt.figure(figsize=(10,5))
    n, bin_edges, patches = plt.hist(flattened_values, bins=bins, color='blue', alpha=0.7)
    
    # Highlight the bars with specific pixel intensities, if required
    if highlight_value is not None:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= highlight_value < bin_edges[i + 1]:
                patches[i].set_facecolor('red')
    
    # Label the counts above the bars
    for i in range(len(patches)):
        height = patches[i].get_height()
        plt.text(patches[i].get_x() + patches[i].get_width() / 2., height + 0.5,
                 '%d' % int(height), ha='center', va='bottom')

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    
    
    
def plot_binary_matrix(binary_matrix, title="Binary Representation"):
    """Plot the binary matrix."""
    plt.imshow(binary_matrix, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
def create_binary_matrix(matrix, threshold):
    """Return a binary matrix where all values equal to the threshold are 1 and others are 0."""
    return (matrix == threshold).astype(int)


def detect_intersection2(large_matrix, target_sum=1000, stride=1, pooling_size=2):
    """
    Detect occurrences of a specific pattern (sum of elements) in a larger matrix.
    
    Args:
        large_matrix: The 2D array where patterns should be detected.
        target_sum: The target sum to look for.
        stride: The number of positions to skip while sliding.
        pooling_size: Size of the smaller matrix (pattern) to check for the target sum.
    
    Returns:
        A matrix of the same size as large_matrix, with detected patterns highlighted.
    """
    result_matrix = np.zeros_like(large_matrix)
    
    rows, cols = large_matrix.shape
    for i in range(0, rows - pooling_size + 1, stride):
        for j in range(0, cols - pooling_size + 1, stride):
            window = large_matrix[i:i+pooling_size, j:j+pooling_size]
            if np.sum(window) == target_sum:
                result_matrix[i:i+pooling_size, j:j+pooling_size] = 1
                
    return result_matrix


def detect_intersection(large_matrix, stride=1):
    """
    Detect occurrences of special 3x3 patterns in a larger matrix.
    
    Args:
        large_matrix: The 2D array where patterns should be detected.
        stride: The number of positions to skip while sliding.
    
    Returns:
        A matrix of the same size as large_matrix, with detected patterns highlighted.
    """
    filters = generate_matrices()
    
    # Convert 250 and 500 in large_matrix to 1, and 0 remains 0
    processed_large_matrix = np.where((large_matrix == 250) | (large_matrix == 500), 1, 0)

    result_matrix = np.zeros_like(large_matrix)
    rows, cols = processed_large_matrix.shape
    
    for filter_matrix in filters:
        for i in range(0, rows - 2, stride):  # Subtracting 2 because we're using 3x3 filters
            for j in range(0, cols - 2, stride):
                window = processed_large_matrix[i:i+3, j:j+3]  # 3x3 window
                if np.array_equal(window, filter_matrix):
                    result_matrix[i:i+3, j:j+3] = 1

    return result_matrix





def plot_matrix(matrix, title="Matrix"):
    plt.imshow(matrix, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
    
    
# def detect_filled_matrices(large_matrix):
#     """
#     Detect occurrences of smaller matrices filled with ones in a larger matrix.
    
#     Args:
#         large_matrix: The 2D array where patterns should be detected.
    
#     Returns:
#         A matrix of the same size as large_matrix, with detected patterns highlighted.
#     """
#     filters = [
#         np.ones((2, 2)),
#         np.ones((2, 3)),
#         np.ones((3, 2))]
#     #]
    
#     result_matrix = np.zeros_like(large_matrix)
#     rows, cols = large_matrix.shape
    
#     for filter_matrix in filters:
#         f_rows, f_cols = filter_matrix.shape
#         for i in range(rows - f_rows + 1):
#             for j in range(cols - f_cols + 1):
#                 window = large_matrix[i:i+f_rows, j:j+f_cols]
#                 if np.array_equal(window, filter_matrix):
#                     result_matrix[i:i+f_rows, j:j+f_cols] = 1
                    
#     return result_matrix    


def detect_filled_matrices(large_matrix):
    """
    Detect occurrences of smaller matrices filled with ones in a larger matrix.
    
    Args:
        large_matrix: The 2D array where patterns should be detected.
    
    Returns:
        A matrix of the same size as large_matrix, with detected patterns highlighted.
    """
    filters = generate_matrices()

    # Convert 250 and 500 in large_matrix to 1, and 0 remains 0
    processed_large_matrix = np.where((large_matrix == 250) | (large_matrix == 500), 1, 0)

    result_matrix = np.zeros_like(large_matrix)
    rows, cols = processed_large_matrix.shape
    
    for filter_matrix in filters:
        f_rows, f_cols = filter_matrix.shape
        for i in range(rows - f_rows + 1):
            for j in range(cols - f_cols + 1):
                window = processed_large_matrix[i:i+f_rows, j:j+f_cols]
                if np.array_equal(window, filter_matrix):
                    result_matrix[i:i+f_rows, j:j+f_cols] = 1
                    
    return result_matrix    




def generate_matrices2():
    matrices = []

    # Base configuration: Central column as all 1's
    base_col = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    
    for i in range(3):
        for j in range(3):
            matrix = base_col.copy()
            matrix[i][0] = 1
            matrix[j][2] = 1
            matrices.append(matrix)

    # Base configuration: Central row as all 1's
    base_row = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    
    for i in range(3):
        for j in range(3):
            matrix = base_row.copy()
            matrix[0][i] = 1
            matrix[2][j] = 1
            matrices.append(matrix)

    return matrices



def generate_matrices():
    matrices = []

    matrix1=np.array([
        [1,0,0],
        [1,1,1],
        [0,1,0]
    ])     
    
    matrix2=np.array([
       [0,0,1],
       [1,1,1],
       [0,1,0]
   ] )  
    
    matrix3=np.array([
       [0,1,0],
       [1,1,1],
       [1,0,0]
   ] ) 
    
    matrix4=np.array([
        [0,1,0],
        [1,1,1],
        [0,0,1]
    ] ) 
   
    
    matrix5=np.array([
        [1,1,0],
        [1,1,1],
        [0,0,1]
    ] ) 
    
    matrix6=np.array([
        [0,1,1],
        [1,1,1],
        [1,0,0]
    ] ) 
    
    matrix7=np.array([
        [1,0,0],
        [1,1,1],
        [0,1,1]
    ] ) 
    
    matrix8=np.array([
        [0,0,1],
        [1,1,1],
        [1,1,0]
    ] ) 
    
    
    matrix9=np.array([
        [1,1,0],
        [1,1,1],
        [1,1,0]
    ] ) 
    
    
    matrix10=np.array([
        [0,1,1],
        [1,1,1],
        [1,1,0]
    ] ) 
    
    
    
    
    matrices=[matrix1,matrix2,matrix3,matrix4,matrix5,matrix6,matrix7,matrix8,
              matrix1.T,matrix2.T,matrix3.T,matrix4.T,matrix5.T,matrix6.T,matrix7.T,matrix8.T
              ]


    return matrices


def extract_submatrices(matrix, size=3, stride=1):
    """
    Extracts all submatrices of given size from the matrix with the specified stride.

    Args:
    - matrix (np.array): The input 2D array.
    - size (int, optional): Size of the submatrix. Default is 3.
    - stride (int, optional): Stride for sliding window. Default is 1.

    Returns:
    - list of np.array: List of submatrices.
    """
    submatrices = []
    for i in range(0, matrix.shape[0] - size + 1, stride):
        for j in range(0, matrix.shape[1] - size + 1, stride):
            submatrices.append(matrix[i:i+size, j:j+size])
    return submatrices

def calculate_sums(submatrices):
    """
    Calculate the sum of each submatrix.

    Args:
    - submatrices (list of np.array): List of submatrices.

    Returns:
    - list of int: List of sums of each submatrix.
    """
    return [np.sum(submatrix) for submatrix in submatrices]

def plot_histogram(sums):
    """
    Plots a histogram of the sums.

    Args:
    - sums (list of int): List of sums.
    """
    plt.hist(sums, bins=50, edgecolor='black')
    plt.xlabel('Sum')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sums of 3x3 Matrices')
    plt.show()

def main(matrix, stride=1):
    submatrices = extract_submatrices(matrix, size=3, stride=stride)
    sums = calculate_sums(submatrices)
    plot_histogram(sums)


def process_submatrices(matrix, stride=1):
    """
    Modifies the input matrix in-place.
    For each 3x3 submatrix:
    - The center element (i.e., element (1,1)) is set to the sum of the submatrix.
    - All other elements of the submatrix are set to 0.
    """
    rows, cols = matrix.shape

    for i in range(0, rows - 2, stride):
        for j in range(0, cols - 2, stride):
            # Extract submatrix
            submatrix = matrix[i:i+3, j:j+3]
            
            # Calculate sum and set the rest to zero
            sub_sum = np.sum(submatrix)
            submatrix.fill(0)
            submatrix[1, 1] = sub_sum
            
            # Update the main matrix
            matrix[i:i+3, j:j+3] = submatrix

    return matrix



def get_top_values(matrix):
    # Flatten the matrix and get unique values and their counts
    values, counts = np.unique(matrix, return_counts=True)
    
    # Sort by counts
    sorted_indices = np.argsort(-counts)
    values = values[sorted_indices]
    counts = counts[sorted_indices]
    
    return values, counts

def display_in_table(values, counts):
    df = pd.DataFrame({
        'Value': values,
        'Count': counts
    })
    # Display the table
    print(df)
    return df
    
    
    
    
    