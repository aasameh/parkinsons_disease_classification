import numpy as np
import os
import glob

# Set the folder where your dataset is saved
folder_path = r"C:\college\6th\PBLx\refactor\watch_incoming"

# Find all .npy files inside the folder
search_pattern = os.path.join(folder_path, "*.npy")
all_files = glob.glob(search_pattern)

# Check if the folder has files
if len(all_files) == 0:
    print(f"No files found in '{folder_path}'!")
else:
    # Sort files so they appear in the order they were created
    all_files.sort()
    
    print(f"Found {len(all_files)} files. Starting to read...\n")
    
    # Loop through each file and print its data
    for index, file_path in enumerate(all_files, start=1):
        # Load the data
        data = np.load(file_path)
        
        # Get just the file name (without the folder path)
        file_name = os.path.basename(file_path)
        
        # Print the header for this reading
        print("========================================")
        print(f"Reading {index}")
        print(f"File: {file_name}")
        print(f"Total Rows: {data.shape[0]}")
        print("========================================")
        
        # Print the actual data
        print(data)
        print("\n\n")  # Add some space before the next file