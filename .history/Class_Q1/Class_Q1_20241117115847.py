import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

#     This code performs the following tasks (OOP version):
#     1. Reads an image and extracts the red channel (I1).
#     2. Permutes I1 and displays the result.
#     3. Presents the histogram of I1.
#     4. Calculates the entropy of I1.
#     5. Computes the mutual information between I1 and its left-neighbors.
#     6. Computes the mutual information between the permuted I1 and its left-neighbors.
#     7. Displays the results.

class ImageAnalysis:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Failed to load image.")
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.I1 = self.image_rgb[:, :, 0]  # Extract the red channel
        self.left_neighbors = np.roll(self.image, shift=1, axis=1)  # Shift right by 1
        self.left_neighbors_hist = np.histogram(self.left_neighbors, bins=256, range=(0, 256), density=True)  # Normalized
        self.I1_hist, _ = np.histogram(self.I1, bins=256, range=(0, 256), density=True)                        # Normalized
        self.I1_entropy = self.calc_entropy(self.I1_hist)
        self.I1_left_neighbors_mui = self.calc_mutual_info(self.I1_hist, self.left_neighbors_hist)
        self.I1_permuted = self.permutate(self.I1)
        self.I1_permuted_hist = np.histogram(self.I1_permuted, bins=256, range=(0, 256), density=True)                        # Normalized
        self.I1_permuted_left_neighbors_mui = self.calc_mutual_info(self.I1_permuted_hist, self.left_neighbors_hist)
    def calc_entropy(hist):
       hist = hist[hist > 0]  # Remove zero entries
       return -np.sum(hist * np.log2(hist)) 
        
    def calc_mutual_info(hist_A, hist_B):
        joint_hist = np.histogram2d(hist_A, hist_B, bins=256, range=(0, 256), density=True) # Normalized
        mui = 0
        for i in range(256):
            for j in range(256):
                if joint_hist[i,j] > 0:
                    mui += joint_hist[i,j] * np.log2(joint_hist[i,j] / (hist_A[i] * hist_B[j]))
        return mui
    
    def permutate(img):
        # Permute I1 (random shuffle of pixel values)
        img_permuted = img.flatten()  # Flatten the image to 1D array (vector)
        np.random.shuffle(img_permuted)  # Shuffle the pixel values
        return img_permuted.reshape(img.shape) # Reshape back to 2D image

        
my_image = ImageAnalysis("Class_Q1/Q1_Image.jpg")
# Show I1
plt.imshow(my_image.I1, cmap='gray')
plt.title("Original I1 (Red Channel)")
plt.axis('off')
plt.show()

# Show permuted I1
plt.imshow(my_image.I1_permuted, cmap='gray')
plt.title("Permuted I1")
plt.axis('off')
plt.show()

# Show histogram of I1
plt.hist(my_image.I1.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.title("Histogram of I1")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Display results
print(f"Entropy of I1: {my_image.I1_entropy:.4f}")
print(f"Mutual Information of I1 with left-neighbors: {my_image.I1_left_neighbors_mui:.4f}")
print(f"Mutual Information of Permuted I1 with left-neighbors: {my_image.I1_permuted_left_neighbors_mui:.4f}")



# def analyze_image(image_path):
#     """

#     """
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Failed to load image.")
#         return

#     # Convert to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Extract the red channel (I1)
#     I1 = image_rgb[:, :, 0]

#     # Permute I1 (random shuffle of pixel values)
#     I1_permuted = I1.flatten()  # Flatten the image to 1D array (vector)
#     np.random.shuffle(I1_permuted)  # Shuffle the pixel values
#     I1_permuted = I1_permuted.reshape(I1.shape) # Reshape back to 2D image

#     # Display the original I1 and the permuted I1
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(I1, cmap='gray')
#     axes[0].set_title("Original I1 (Red Channel)")
#     axes[0].axis('off')

#     axes[1].imshow(I1_permuted, cmap='gray')
#     axes[1].set_title("Permuted I1")
#     axes[1].axis('off')
#     plt.show()

#     # Display histogram of I1
#     plt.hist(I1.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
#     plt.title("Histogram of I1")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.show()

#     # Calculate entropy of I1
#     hist = np.histogram(I1, bins=256, range=(0, 256))[0]
#     hist_prob = hist / np.sum(hist)                     # Compute probability distribution
#     I1_entropy = entropy(hist_prob, base=2)             # Compute entropy
#     print(f"Entropy of I1: {I1_entropy:.4f}")       # Display entropy

#     # Calculate mutual information function
#     def calculate_mutual_information(image):
#         left_neighbors = np.roll(image, shift=-1, axis=1)  # Shift right by 1
#         image_flat = image.flatten()                        # Flatten the image
#         left_flat = left_neighbors.flatten()                # Flatten the left neighbors
#         mi = mutual_info_score(image_flat, left_flat)           # Compute mutual information
#         return mi

#     # Calculate mutual information for original and permuted I1
#     mi_original = calculate_mutual_information(I1)
#     mi_permuted = calculate_mutual_information(I1_permuted)

#     print(f"Mutual Information of I1 with left-neighbors: {mi_original:.4f}")
#     print(f"Mutual Information of Permuted I1 with left-neighbors: {mi_permuted:.4f}")

#     def entropy

# # Call the function
# Image_path = "Class_Q1/Q1_Image.jpg"
# analyze_image(Image_path)



