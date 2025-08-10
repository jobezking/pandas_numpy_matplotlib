import numpy as np

# Step 1: create list of numbers
my_list = [5, 10, 15, 20]

# Step 2: convert list into NumPy array
my_array = np.array(my_list)

# Step 3: Print array to see result
print("NumPy Array: ", my_array)

# Multi-dimensional
# Step 1: create nested list
a = [1,2,3]
b = [4,5,6]
c = [7,8,9]
nested_list = [[a,b,c],[a,b,c],[a,b,c]]
matrix = np.array(nested_list)
print(matrix)

#array inspection
print("Shape of matrix: ", matrix.shape)    # dimensions (number of rows)
print("Total elements: ", matrix.size)      # Number of elements
print("Data Type: ", matrix.dtype)          # Type of elements

# array creation
zeroes = np.zeros((3,4,5))
ones = np.ones((2,5))
randoms = np.random.rand(3,3)
print(zeroes)
print(ones)
print(randoms)