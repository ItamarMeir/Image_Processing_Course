import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        self.left_neighbors = np.roll(self.I1, shift=1, axis=1)  # Shift right by 1

        # self.left_neighbors_hist = np.histogram(self.left_neighbors, bins=256, range=(0, 256), density=True)  # Normalized
        self.I1_hist, _ = np.histogram(self.I1, bins=256, range=(0, 256), density=True)                        # Normalized
        self.I1_entropy = self.calc_entropy(self.I1_hist)
        self.I1_left_neighbors_mui = self.calc_mutual_info(self.I1, self.left_neighbors)
        self.I1_permuted = self.permutate(self.I1)
        self.I1_permuted_hist = np.histogram(self.I1_permuted, bins=256, range=(0, 256), density=True)                        # Normalized
        self.I1_permuted_left_neighbors_mui = self.calc_mutual_info(self.I1_permuted, self.left_neighbors)
    
    def calc_entropy(self, hist):
       hist = hist[hist > 0]  # Remove zero entries
       return -np.sum(hist * np.log2(hist)) 
        
    def calc_mutual_info(self, data_A, data_B):
        joint_hist, x_edges, y_edges = np.histogram2d(data_A.ravel(), data_B.ravel(), bins=256, range=[[0, 256], [0, 256]], density=True)  # Normalized
        prob_A = np.sum(joint_hist, axis=1)
        prob_B = np.sum(joint_hist, axis=0)
        mui = 0
        for i in range(256):
            for j in range(256):
                if joint_hist[i, j] > 0:
                    mui += joint_hist[i, j] * np.log2(joint_hist[i, j] / (prob_A[i] * prob_B[j]))
        return mui
    
    def permutate(self, img):
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







