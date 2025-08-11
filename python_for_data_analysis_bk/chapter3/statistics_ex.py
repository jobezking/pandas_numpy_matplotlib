import numpy as np

def summarize_data(arr):
    print(f"Mean: {np.mean(arr)}") 
    print(f"Median: {np.median(arr)}")  
    print(f"Variance: {np.var(arr)}")  
    print(f"Standard Deviation: {np.std(arr)}") 
    print(f"Min: {np.min(arr)}")  
    print(f"Max: {np.max(arr)}")  
    print(f"Range: {np.max(arr) - np.min(arr)}")
    print(f"25th Percentile (Q1): {np.percentile(arr, 25)}")  
    print(f"75th Percentile (Q3): {np.percentile(arr, 75)}")

summarize_data([5, 10, 15, 20, 25])