
import os
import pandas as pd
import cv2

# Define the paths
csv_folder = r"E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset0\dots_csv"
image_folder = r"E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset0\images"
output_folder = r'E:\FASTRCNN\FASTRCNN\dataset\000000origin_data\new syntetic dataset0\im&dot'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all CSV files
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(os.path.join(csv_folder, csv_file))

    # Extract the base name (without extension) to find the corresponding image
    base_name = os.path.splitext(csv_file)[0]
    image_path = os.path.join(image_folder, base_name + '.jpg')

    # Read the image
    image = cv2.imread(image_path)

    # Plot the points on the image
    for index, row in df.iterrows():
        x, y = int(row['cx']), int(row['cy'])
        cv2.circle(image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    # Save the modified image to the output folder
    output_path = os.path.join(output_folder, base_name + '.jpg')
    cv2.imwrite(output_path, image)

print('All images have been processed and saved successfully.')
