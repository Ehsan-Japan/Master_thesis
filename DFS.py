# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 03:55:45 2023

@author: ehsan
"""
import numpy as np

def dfs(matrix, x, y, target_x, visited):
    if x < 0 or x >= 3 or y < 0 or y >= 3 or visited[x][y] or matrix[x][y] == 0:
        return False
    if x == target_x:
        return True
    visited[x][y] = True
    # Explore neighbors (up, down, left, right)
    if (dfs(matrix, x+1, y, target_x, visited) or 
        dfs(matrix, x-1, y, target_x, visited) or 
        dfs(matrix, x, y+1, target_x, visited) or 
        dfs(matrix, x, y-1, target_x, visited)):
        return True
    return False

def can_cross(matrix, direction):
    if direction == 'horizontal':
        for i in range(3):
            if matrix[i][0] == 1:
                visited = [[False for _ in range(3)] for _ in range(3)]
                if dfs(matrix, i, 0, i, visited):
                    return True
    elif direction == 'vertical':
        for i in range(3):
            if matrix[0][i] == 1:
                visited = [[False for _ in range(3)] for _ in range(3)]
                if dfs(matrix, 0, i, 2, visited):
                    return True
    return False

def generate_all_matrices():
    solutions = []
    for i in range(512):  # 2^9 = 512 possible 3x3 matrices with 0s and 1s
        matrix = np.array([(i >> j) & 1 for j in reversed(range(9))]).reshape(3, 3)
        if can_cross(matrix, 'horizontal') or can_cross(matrix, 'vertical'):
            solutions.append(matrix)
    return solutions

solutions = generate_all_matrices()
for idx, matrix in enumerate(solutions, 1):
    print(f"Solution {idx}:\n", matrix, "\n")