import matplotlib.pyplot as plt
import numpy as np


# Epochs and precision values
epochs = list(range(1, 101))  
# Read the text file
with open("D:\\Youtube_MaskRCNN\\newMetrics.txt", 'r') as file:
    lines = file.readlines()

# Initialize an empty list to store precision values
precision_list = []

# Iterate over each line in the text file
for line in lines:
    # Split the line to extract precision value (assuming precision is after 'Precision - ')
    precision = float(line.split('precision - ')[1].split(',')[0])
    # Append the precision value to the precision list
    precision_list.append(precision)

print(precision_list)


# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, precision, marker='o', linestyle='-', color='black', label='Precision')
plt.title('Precision vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.xticks(range(0, 101, 5))
plt.grid(True)
plt.show()
