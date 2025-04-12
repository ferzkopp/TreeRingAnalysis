"""
AnalyzeTreeImage.py

This script processes an image of tree rings to estimate the age of a tree by analyzing circular patterns and brightness variations. 
It applies image processing techniques, detects circular features, and performs statistical analysis on the detected markers.

Andreas Schiffler, April 2025

Modules:
    - cv2: For image processing and visualization.
    - numpy: For numerical computations.
    - scipy.signal: For peak detection in brightness values.
    - scipy.stats: For statistical analysis (e.g., mode, Gaussian fit).
    - matplotlib.pyplot: For plotting and visualizing data.
    - random: For sampling.
    - scipy.optimize: For optimization tasks (e.g., finding the best center).
Configuration:
    - image_filename (str): Path to the input image file.
    - show_visuals (bool): Whether to display intermediate visualizations.
    - min_radius (int): Minimum radius from the center to analyze.
    - cutoff (float): Cutoff frequency for time series filtering.
    - prominence (int): Prominence threshold for peak detection.
Functions:
    - cost_function(center, edges): Computes the cost for evaluating the center of circular features.
    - find_best_center(edges, initial_center): Optimizes the center of circular features using gradient descent.
Workflow:
    1. Load and preprocess the image (e.g., Gaussian blur, HSV conversion).
    2. Perform color analysis and binarize the image based on brightness thresholds.
    3. Detect the center of circular features using optimization.
    4. Analyze pixel brightness along radial lines from the detected center.
    5. Apply frequency filtering and detect peaks/troughs in brightness values.
    6. Visualize and save results, including marker overlays and statistical plots.
    7. Estimate the tree's age based on statistical metrics (mean, mode, max) of the set of detected peaks/troughs.
Outputs:
    - Binarized image with detected circular features.
    - Image with markers for peaks and troughs.
    - Histogram of marker series with Gaussian fit.
    - Estimated age range of the tree.
Note:
    Ensure the input image is properly formatted and contains visible tree rings for accurate analysis.
"""

# Import necessary libraries
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.stats import mode
import matplotlib.pyplot as plt
from random import sample
from scipy.optimize import minimize
import os

# --- Configuration

show_visuals = False  # Set to True to show images and charts, False to skip showing them

image_filename = "Sample1/Tree.png"  # Path to the image file
binarization_threshold_ptile = 30  # Adjust percentile for binarization threshold
cutoff = 0.045  # Adjust the cutoff frequency for time series filtering as needed
prominence = 4  # Adjust prominence for peak detection as needed
min_radius_percent = 1  # Minimum radius from center to analyze (as percent of image size)

# image_filename = "Sample2/Tree.jpg"  # other sample image file
# binarization_threshold_ptile = 40
# cutoff = 0.1  
# prominence = 4 
# min_radius_percent = 1

# image_filename = "Sample3/Tree.png"  # another sample image file
# binarization_threshold_ptile = 45
# cutoff = 0.08  
# prominence = 4 
# min_radius_percent = 1  

# -----

print("Loading the image...")
image = cv2.imread(image_filename)
if image is None:
    raise FileNotFoundError(f"Could not load the image. Please check the filename: {image_filename}")
image_extension = os.path.splitext(image_filename)[1]  # Extract the file extension

# Scale the image if the width is under 3000 pixels
if image.shape[1] < 3000:
    print("Scaling the image to 3000 pixels width while maintaining aspect ratio...")
    scale_factor = 3000 / image.shape[1]
    new_width = 3000
    new_height = int(image.shape[0] * scale_factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(f"Image scaled to {new_width}x{new_height} pixels.")

# Apply Gaussian blur multiple times to reduce small details
print("Applying Gaussian blur to the image...")
blurred_image = image.copy()
blurred_image = cv2.GaussianBlur(blurred_image, (7, 7), 0)
blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), 0)
blurred_image = cv2.GaussianBlur(blurred_image, (3, 3), 0)

# Perform color analysis on the center quarter of the image
print("Analyzing the colors in the center quarter of the image...")
height, width, _ = blurred_image.shape
center_quarter = blurred_image[height // 4: 3 * height // 4, width // 4: 3 * width // 4]

# Draw a bounding box around the center quarter on a copy of the blurred image
blurred_image_with_box = blurred_image.copy()
cv2.rectangle(
    blurred_image_with_box,
    (width // 4, height // 4),
    (3 * width // 4, 3 * height // 4),
    (0, 255, 0),  # Green color for the bounding box
    2  # Thickness of the rectangle
)

# Display the image with the bounding box
if show_visuals:
    cv2.imshow("Center Quarter Bounding Box", blurred_image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save the image with the bounding box
bounding_box_output_filename = image_filename.replace(image_extension, "-CenterQuarterBoundingBox" + image_extension)
cv2.imwrite(bounding_box_output_filename, blurred_image_with_box)
print(f"Image with center quarter bounding box saved as {bounding_box_output_filename}")

# Convert the center quarter to HSV for better color segmentation
hsv_center = cv2.cvtColor(center_quarter, cv2.COLOR_BGR2HSV)

# Calculate histograms for each channel
h_hist = cv2.calcHist([hsv_center], [0], None, [180], [0, 180])
s_hist = cv2.calcHist([hsv_center], [1], None, [256], [0, 256])
v_hist = cv2.calcHist([hsv_center], [2], None, [256], [0, 256])

# Convert the entire image to HSV for better color segmentation
hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

# Determine thresholds for darker parts (tree rings) and brighter parts
dark_threshold = np.percentile(hsv_image[:, :, 2], binarization_threshold_ptile)  # Lower ptile for darker parts
bright_threshold = np.percentile(hsv_image[:, :, 2], 100 - binarization_threshold_ptile)  # Upper ptile for brighter parts

print ("Binarized image using thresholds:")
print(f"Dark threshold: {dark_threshold}, Bright threshold: {bright_threshold}")
# Create masks for the two ranges
dark_mask = hsv_image[:, :, 2] <= dark_threshold
bright_mask = hsv_image[:, :, 2] >= bright_threshold

# Map the ranges to black (dark parts) and white (bright parts)
binary_image = np.zeros_like(hsv_image[:, :, 2], dtype=np.uint8)
binary_image[dark_mask] = 255 
binary_image[bright_mask] = 0

# Save the binary image for visualization
binary_output_filename = image_filename.replace(image_extension, "-Binarized" + image_extension)
cv2.imwrite(binary_output_filename, binary_image)
print(f"Binarized image saved as {binary_output_filename}")

print("Finding a center for circular features in the binarized image...")

# Create a circular mask to keep only the centered circle
height, width = binary_image.shape
center = (width // 2, height // 2)
radius = int((height // 2) * 0.8)  # Shrink the radius by 20%
circular_mask = np.zeros_like(binary_image, dtype=np.uint8)
cv2.circle(circular_mask, center, radius, 255, -1)

# Apply the circular mask to the binary image
binary_image = cv2.bitwise_and(binary_image, circular_mask)

# Cost function to evaluate the center
def cost_function(center, edges):
    x_center, y_center = center
    y_indices, x_indices = np.where(edges > 0)  # Get edge points
    radii = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
    mean_radius = np.mean(radii)
    cost = np.sum((radii - mean_radius)**2)  # Minimize the deviation from the mean radius
    return cost

# Gradient descent to find the best center
def find_best_center(edges, initial_center):
    result = minimize(
        cost_function, 
        initial_center, 
        args=(edges,), 
        method='Powell'
    )
    return result.x

# Find the best center in the binarized image
initial_center = (image.shape[1] // 2, image.shape[0] // 2)  # Start with the image center as the initial guess
best_center = find_best_center(binary_image, initial_center)  # Optimize to find the most accurate center

# Draw the initial center in green and the detected center in red, then draw an arrow between them
binary_gray = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
cv2.circle(binary_gray, (int(initial_center[0]), int(initial_center[1])), 15, (0, 255, 0), -1)  # Initial center
cv2.circle(binary_gray, (int(best_center[0]), int(best_center[1])), 15, (0, 0, 255), -1)  # Detected center
cv2.arrowedLine(binary_gray, (int(initial_center[0]), int(initial_center[1])), 
                (int(best_center[0]), int(best_center[1])), (0, 255, 255), 2, tipLength=0.2)  # Arrow
print(f"Best center: ({int(best_center[0])}, {int(best_center[1])})")

# Save the image with the detected circle
circle_output_filename = image_filename.replace(image_extension, "-DetectedCenter" + image_extension)
cv2.imwrite(circle_output_filename, binary_gray)
print(f"Detected center image saved as {circle_output_filename}")

print("Converting the blurred image to grayscale...")
gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# print("Enhancing the contrast of the grayscale image...")
gray = cv2.equalizeHist(gray)

# Save the enhanced grayscale image used for further analysis
gray_output_filename = image_filename.replace(image_extension, "-BlurredGrayForAnalysis" + image_extension)
cv2.imwrite(gray_output_filename, gray)

print("Analyzing pixel brightness along radial lines from center...")

# Initialize variables
center_x, center_y = int(best_center[0]), int(best_center[1])  # Use the detected center of the tree rings
min_radius = int(min_radius_percent / 100 * min(image.shape[0], image.shape[1]))  # Minimum radius in pixels
max_radius = int(max(image.shape[0]/2, image.shape[1]/2)) # Maximum radius in pixels
radii = range(min_radius, max_radius)  # Radii to analyze
angles = np.deg2rad(np.arange(0, 360, 1))  # Angles in radians 
marker_series = []

# Iterate over angles
for theta in angles:
    brightness_values = []
    # Iterate over radii
    for r in radii:
        # Calculate the coordinates of the point along the ray
        x = int(center_x + r * np.cos(theta))
        y = int(center_y + r * np.sin(theta))
        # Ensure the coordinates are within bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            brightness_values.append(gray[y, x])
            # Calculate the starting point of the ray at min_radius
            start_x = int(center_x + min_radius * np.cos(theta))
            start_y = int(center_y + min_radius * np.sin(theta))
            # Draw the ray on the image from min_radius to the current point
            cv2.line(image, (start_x, start_y), (x, y), (128, 128, 128), 1)
    
    # Apply a frequency filter with dynamic frequency response
    fft_values = np.fft.fft(brightness_values)
    frequencies = np.fft.fftfreq(len(brightness_values))
    
    # Create a dynamic cutoff frequency response
    dynamic_cutoff = cutoff + (cutoff * 2) * (frequencies / max(frequencies))
    
    # Plot the dynamic cutoff filter response only for the first iteration
    if theta == 0 and show_visuals:
        plt.figure()
        plt.plot(frequencies, dynamic_cutoff, label="Dynamic Cutoff Response", color='blue')
        plt.title("Dynamic Cutoff Frequency Response")
        plt.xlabel("Frequency")
        plt.ylabel("Cutoff Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot as an image
        cutoff_plot_filename = image_filename.replace(image_extension, "-DynamicCutoffResponse" + image_extension)
        plt.savefig(cutoff_plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    # Apply the dynamic cutoff to remove high-frequency components
    fft_values[np.abs(frequencies) > dynamic_cutoff] = 0
    
    # Perform the inverse FFT to get the filtered brightness values
    smoothed_brightness_values = np.fft.ifft(fft_values).real

    # Detect troughs in the smoothed brightness values
    troughs, _ = find_peaks(-smoothed_brightness_values, prominence=prominence)
    
    # Detect peaks in the smoothed brightness values
    peaks, _ = find_peaks(smoothed_brightness_values, prominence=prominence)
    
    # Display the smoothed brightness values in a line chart for specific angles
    if theta in [0, np.pi / 4, np.pi / 2] and show_visuals:  # Show the chart for angles 0°, 45°, and 90° only
        if theta == 0:
            plt.figure(figsize=(15, 12))  # Set the figure size
        subplot_index = {0: 1, np.pi / 4: 2, np.pi / 2: 3}[theta]  # Map angles to subplot indices
        plt.subplot(3, 1, subplot_index)  # Create a stacked subplot
        plt.plot(brightness_values, label="Original Brightness")
        plt.plot(smoothed_brightness_values, label="Smoothed Brightness", linestyle='--')
        plt.scatter(peaks, smoothed_brightness_values[peaks], color='blue', label="Peaks", zorder=5)
        plt.scatter(troughs, smoothed_brightness_values[troughs], color='orange', label="Troughs", zorder=5)
        plt.title(f"Brightness Values for Angle {np.rad2deg(theta):.1f}°")
        plt.xlabel("Radius")
        plt.ylabel("Brightness")
        plt.legend()
        
        if theta == np.pi / 2:  # Show the combined plot after the third chart
            plt.tight_layout()
            output_histogram_filename = image_filename.replace(image_extension, "-MarkerAnalysis" + image_extension)
            plt.savefig(output_histogram_filename, dpi=300, bbox_inches='tight')
            plt.show()
        
    # Draw the peaks onto the image
    for peak in peaks:
        peak_x = int(center_x + radii[peak] * np.cos(theta))
        peak_y = int(center_y + radii[peak] * np.sin(theta))
        if 0 <= peak_x < image.shape[1] and 0 <= peak_y < image.shape[0]:
            cv2.circle(image, (peak_x, peak_y), 5, (255, 0, 0), -1)  # Draw a small blue circle at the peak
    
    # Draw the troughs onto the image
    for trough in troughs:
        trough_x = int(center_x + radii[trough] * np.cos(theta))
        trough_y = int(center_y + radii[trough] * np.sin(theta))
        if 0 <= trough_x < image.shape[1] and 0 <= trough_y < image.shape[0]:
            cv2.circle(image, (trough_x, trough_y), 5, (0, 165, 255), -1)  # Draw a small orange circle at the trough
    
    # Append the number of peaks and troughs in this ray to the series
    marker_series.append(len(peaks))
    marker_series.append(len(troughs))

# Save the non-resized image with markers as a new file
output_image_filename = image_filename.replace(image_extension, "-MarkerOverlay" + image_extension)
cv2.imwrite(output_image_filename, image)
print(f"Image with ring-markers overlay saved as {output_image_filename}")

# Save the upper right third section of the image with a 200-pixel offset from the edge as a new file
height, width, _ = image.shape
upper_right_section = image[0:height // 3, width // 3 + 200:width]
output_section_filename = image_filename.replace(image_extension, "-MarkerOverlaySection" + image_extension)
cv2.imwrite(output_section_filename, upper_right_section)
print(f"Section of the image with ring-markers overlay saved as {output_section_filename}")

if show_visuals:
    # Show the image
    cv2.imshow("Tree Rings with Markers (upper right)", upper_right_section)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Calculate a histogram of the marker_series
plt.figure()
counts, bins, _ = plt.hist(marker_series, bins=30, color='blue', edgecolor='black', alpha=0.7, density=True)
plt.title("Histogram of Marker Series")
plt.xlabel("Number of Ring Markers")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust x-axis labels to align with the bins
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.xticks(bin_centers, labels=[f"{int(b) + 1}" for b in bins[:-1]], rotation=45, fontsize='small')

# Fit a Gaussian to the histogram
bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
mean, std = norm.fit(marker_series)  # Fit a Gaussian distribution
gaussian_fit = norm.pdf(bin_centers, mean, std)  # Gaussian curve

# Calculate the mode of the marker series
mode_value, mode_count = mode(marker_series)
print(f"The mode of the marker series is {mode_value} with a count of {mode_count}.")

# Calculate the maximum value of the marker series
max_value = max(marker_series)
print(f"The maximum value of the marker series is {max_value}.")

# Plot the Gaussian fit as an overlay
plt.plot(bin_centers, gaussian_fit, color='red', linestyle='--', label=f'Gaussian Fit, Mean: {int(mean)}')

# Add the mode value as a vertical line to the plot
plt.axvline(mode_value, color='green', linestyle='-.', label=f'Mode: {mode_value}')

# Add the max value as a vertical line to the plot
plt.axvline(max_value, color='purple', linestyle=':', label=f'Max: {max_value}')

plt.legend()

# Save the plot as an image
output_histogram_filename = image_filename.replace(image_extension, "-MarkerHistogram" + image_extension)
plt.savefig(output_histogram_filename, dpi=300, bbox_inches='tight')
print(f"Histogram of marker series saved as {output_histogram_filename}")

if show_visuals:
    # Show the plot
    plt.show()

# Calculate the estimated age of the tree using the Gaussian mean, mode, and max value
estimated_age_mean = int(mean)
estimated_age_mode = int(mode_value)
estimated_age_max = int(max_value)
estimated_age_range = (min(estimated_age_mean, estimated_age_mode, estimated_age_max), 
                       max(estimated_age_mean, estimated_age_mode, estimated_age_max))

print(f"The estimated age of the tree is between {estimated_age_range[0]} and {estimated_age_range[1]} years.")
