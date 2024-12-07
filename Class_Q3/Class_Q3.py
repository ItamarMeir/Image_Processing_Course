import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_shape(image, show_histogram=False):
    """
    Detects whether the given image contains a circle, square, or triangle 
    by dividing the edge orientations into 4 bins:
    Bin 1: 0 to 45 degrees
    Bin 2: 45 to 90 degrees
    Bin 3: 90 to 135 degrees
    Bin 4: 135 to 180 degrees

    If 2 bins have significant edges -> square
    If 3 bins have significant edges -> triangle
    Else -> circle
    """
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)

    # Sobel filters
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)

    # Compute gradients
    Gx = cv2.filter2D(gray, -1, kernel_x)
    Gy = cv2.filter2D(gray, -1, kernel_y)

    # Compute gradient magnitude and angle
    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx)  # radians from -pi to pi

    # Convert angles to degrees [0,180)
    angle_deg = np.degrees(angle)
    angle_deg[angle_deg < 0] += 180.0

    # Threshold magnitude to remove noise
    mag_threshold = 50
    valid_mask = magnitude > mag_threshold
    valid_angles = angle_deg[valid_mask]

    # Divide into 4 bins:
    # 0-45°, 45-90°, 90-135°, 135-180°
    bin_counts = [0, 0, 0, 0]

    for a in valid_angles:
        if 0 <= a < 45:
            bin_counts[0] += 1
        elif 45 <= a < 90:
            bin_counts[1] += 1
        elif 90 <= a < 135:
            bin_counts[2] += 1
        elif 135 <= a < 180:
            bin_counts[3] += 1

    # Determine which bins are "significant"
    # For example, consider bins significant if they have at least 20% of the max count
    max_count = max(bin_counts) if len(bin_counts) > 0 else 0
    if max_count == 0:
        # No edges detected, assume circle as fallback
        return 'circle'

    threshold = 0.2 * max_count
    significant_bins = sum(1 for c in bin_counts if c > threshold)

    # Decide shape
    if significant_bins == 2:
        shape = 'square'
    elif significant_bins == 3:
        shape = 'triangle'
    else:
        # If 1 or 4 bins are significant, or none,
        # we assume a circle as per instructions
        shape = 'circle'

    print("Detected shape:", shape)

    # Debug visualization
    if show_histogram:
        bin_ranges = ['0-45', '45-90', '90-135', '135-180']
        plt.figure(figsize=(8,4))
        plt.bar(bin_ranges, bin_counts)
        plt.title("4-bin Angle Histogram")
        plt.xlabel("Angle Range (degrees)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

    return shape

statistic_check = False

if statistic_check:
    loops = 50
    counter = 0
    for i in range(loops):
        # Pick a random shape and random 1-3 number, then load the corresponding image
        shape = np.random.choice(['circle', 'square', 'triangle'])
        number = np.random.randint(1, 4)
        image_path = os.path.join('Class_Q3', f"{shape}{number}.png")
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load image, stopping!: {image_path}")
            raise SystemExit
        
        detected_shape = detect_shape(image, show_histogram=False)
        if detected_shape == shape:
            #print("Correctly detected shape!")
            counter += 1

    success_rate = counter / loops * 100
    print(f"Success rate: {success_rate:.2f}%")

else:
    # Pick a random shape and random 1-3 number, then load the corresponding image
    shape = np.random.choice(['circle', 'square', 'triangle'])
    number = np.random.randint(1, 4)
    image_path = os.path.join('Class_Q3', f"{shape}{number}.png")
    image = cv2.imread(image_path)
    print(f"Loaded image: {image_path}")

    if image is None:
        print(f"Could not load image: {image_path}")
        raise SystemExit
    
    detected_shape = detect_shape(image, show_histogram=True)
    if detected_shape == shape:
        print("Correctly detected shape!")
       
