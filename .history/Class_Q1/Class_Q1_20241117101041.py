import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def analyze_image(image_path):
    """
    This function performs the following tasks:
    1. Reads an image and extracts the red channel (I1).
    2. Permutes I1 and displays the result.
    3. Presents the histogram of I1.
    4. Calculates the entropy of I1.
    5. Computes the mutual information between I1 and its left-neighbors.
    6. Computes the mutual information between the permuted I1 and its left-neighbors.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the red channel (I1)
    I1 = image_rgb[:, :, 0]

    # Permute I1 (random shuffle of pixel values)
    I1_permuted = I1.flatten()  # Flatten the image to 1D array (vector)
    np.random.shuffle(I1_permuted)  # Shuffle the pixel values
    I1_permuted = I1_permuted.reshape(I1.shape) # Reshape back to 2D image

    # Display the original I1 and the permuted I1
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(I1, cmap='gray')
    axes[0].set_title("Original I1 (Red Channel)")
    axes[0].axis('off')

    axes[1].imshow(I1_permuted, cmap='gray')
    axes[1].set_title("Permuted I1")
    axes[1].axis('off')
    plt.show()

    # Display histogram of I1
    plt.hist(I1.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title("Histogram of I1")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

    # Calculate entropy of I1
    hist = np.histogram(I1, bins=256, range=(0, 256))[0]
    hist_prob = hist / np.sum(hist)
    I1_entropy = entropy(hist_prob, base=2)
    print(f"Entropy of I1: {I1_entropy:.4f}")

    # Calculate mutual information function
    def calculate_mutual_information(image):
        left_neighbors = np.roll(image, shift=-1, axis=1)  # Shift right by 1
        image_flat = image.flatten()
        left_flat = left_neighbors.flatten()
        mi = mutual_info_score(image_flat, left_flat)
        return mi

    # Calculate mutual information for original and permuted I1
    mi_original = calculate_mutual_information(I1)
    mi_permuted = calculate_mutual_information(I1_permuted)

    print(f"Mutual Information of I1 with left-neighbors: {mi_original:.4f}")
    print(f"Mutual Information of Permuted I1 with left-neighbors: {mi_permuted:.4f}")

# Call the function
Image_path = "Class_Q1\Q1_Image.jpg"
analyze_image(Image_path)



