#!/usr/bin/python3.9

# Instructions to Run the Example Script:
# 1. Execute the script using the command: ./example.py
#    Ensure you have execution permissions. If not, use 'chmod +x example.py' to make it executable.

# 2. Determine the path to your Python interpreter:
#    Use the command 'which python3.x' (replace 'x' with your specific Python version number) to find the path to your Python interpreter.

# 3. Update the script's shebang line:
#    Modify the first line of 'example.py' to match the path of your Python interpreter identified in step 2.
#    For example, if your Python version is 3.9 and 'which python3.9' returns '/usr/bin/python3.9', 
#    then the first line of 'example.py' should be changed to '#!/usr/bin/python3.9' to ensure the script uses the correct Python interpreter.


# Install the following python modules using pip or your package manager of choice.
# This example assumes the use of Python 3.9 and pip. Adjust the command according to your Python version and environment.

# To install the Ultralytics Plus package, which includes the YOLO model and utility functions like render_result.
# sudo -H python3.9 -m pip install ultralyticsplus

# OpenCV package for image processing tasks, necessary if your project involves image manipulation or processing.
# sudo -H python3.9 -m pip install opencv-python-headless  # or just opencv-python if you don't need the headless version

# Pandas for handling and analyzing data in a tabular form.
# sudo -H python3.9 -m pip install pandas

# NumPy for numerical operations on arrays and matrices, which is often used alongside Pandas.
# sudo -H python3.9 -m pip install numpy

# Pillow for image manipulation tasks, especially useful when dealing with the output of render_result which returns PIL.Image objects.
# sudo -H python3.9 -m pip install Pillow

# Glob for file pattern matching, useful for operations like deleting all files of a certain type within a directory.
# Note: glob is part of the Python Standard Library, so you may not need to install it separately.


from ultralyticsplus import YOLO, render_result
import logging
import warnings
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import glob


# Suppress all warnings
warnings.filterwarnings("ignore")

# Create a null handler to suppress all log messages
null_handler = logging.NullHandler()

# Set the root logger's handler to the null handler
logging.getLogger().addHandler(null_handler)

# Get the directory where the script is running
script_dir = os.path.dirname(os.path.abspath(__file__))

# Delete all .jpg files in the current directory (script_dir)
for jpg_file in glob.glob(os.path.join(script_dir, '*.jpg')):
    os.remove(jpg_file)

# Construct the file path for the model
model_path = os.path.join(script_dir, 'best.pt')

# Load the model
model = YOLO(model_path)

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1  # maximum number of detections per image

# Construct file path for test data
data_path = os.path.join(script_dir, 'testdata.parquet')
data = pd.read_parquet(data_path)

# Assuming 'data' is your DataFrame and it contains columns named 'datetime' and 'symbol'
# Select the trading data for a specific day and symbol
desired_date = '2023-05-26'
desired_symbol = 'ZTS'  # Example symbol

# Convert the 'datetime' column to datetime type if it's not already
data['datetime'] = pd.to_datetime(data['datetime'])

# Filter the data for the desired date
days_data = data[data['datetime'].dt.date == pd.to_datetime(desired_date).date()]

# Further filter the data for the desired symbol
day_symbol_data = days_data[days_data['symbol'] == desired_symbol]

# Initialize a sequence number for file naming
sequence_number = 0  

# Continue only if filtered_data is not empty
if not day_symbol_data.empty:
    # Extract 'high' and 'low' values
    high_values = day_symbol_data['high'].values
    low_values = day_symbol_data['low'].values

    # Replace NaN and zero values with the last known good value
    for i in range(len(high_values)):
        if np.isnan(high_values[i]) or high_values[i] == 0:
            high_values[i] = high_values[i - 1] if i > 0 else high_values[i]

    for i in range(len(low_values)):
        if np.isnan(low_values[i]) or low_values[i] == 0:
            low_values[i] = low_values[i - 1] if i > 0 else low_values[i]

    # Normalize the 'high' and 'low' values to the range [0, 1]
    max_value = max(high_values)
    min_value = min(low_values)
    value_range = max_value - min_value
    normalized_high_values = (high_values - min_value) / value_range
    normalized_low_values = (low_values - min_value) / value_range

    # Create an OpenCV Mat with a white background
    width, height = 40, 400  # Adjust the size as needed
    graph = np.zeros((height, width, 3), dtype=np.uint8)
    graph.fill(255)  # Fill with white

    # Calculate scaling factors for the 'high' and 'low' values
    scaled_high_values = (normalized_high_values * (height - 20)).astype(np.float32)
    scaled_low_values = (normalized_low_values * (height - 20)).astype(np.float32)

    # Scale the values to fit within the image height
    scaling_factor = 0.9  # Adjust as needed to fit the graph within the image
    scaled_high_values *= scaling_factor
    scaled_low_values *= scaling_factor

    # Draw the graph lines using OpenCV
    for j in range(1, 391 - 34):
        maxv = 33 if j >= 33 else j

        graph = np.zeros((height, width, 3), dtype=np.uint8)
        graph.fill(255)  # Fill with white

        for i in range(j, j + maxv):
            x = i - j
            pt1 = (x - 1, height - 20 - int(scaled_high_values[i - 1]))
            pt2 = (x, height - 20 - int(scaled_high_values[i]))
            cv2.line(graph, pt1, pt2, (0, 0, 255), 2)  # High values in red

            pt1 = (x - 1, height - 20 - int(scaled_low_values[i - 1]))
            pt2 = (x, height - 20 - int(scaled_low_values[i]))
            cv2.line(graph, pt1, pt2, (0, 255, 0), 2)  # Low values in green
            
        results = model.predict(graph, verbose=False)
        
        render = render_result(model=model, image=graph, result=results[0])
        
        # Assuming 'render' contains the PIL Image returned by render_result
        render_np = np.array(render)  # Convert PIL Image to numpy array

        # Convert RGB to BGR (OpenCV uses BGR color order)
        render_np = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

        # Create the file name with sequency number
        filename = f"graph_{sequence_number}.jpg"
        
        # Figure out the local path
        output_path = os.path.join(script_dir, filename)
        
        # Save your sample
        cv2.imwrite(filename, render_np)
        
        # Increment sequence
        sequence_number += 1

