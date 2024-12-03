import cv2
import numpy as np

def get_pyramids(image):
    """
    Given an input image, returns 3-level Gaussian and Laplacian pyramids.
    
    Parameters:
        image (numpy.ndarray): Input image.
        
    Returns:
        gaussian_pyramid (list): List of images in the Gaussian pyramid.
        laplacian_pyramid (list): List of images in the Laplacian pyramid.
    """
    # Initialize the Gaussian pyramid with the original image
    G = image.copy()
    gaussian_pyramid = [G]
    
    # Build Gaussian pyramid
    for i in range(3):
        G = cv2.pyrDown(G)
        gaussian_pyramid.append(G)
    
    # Build Laplacian pyramid
    laplacian_pyramid = []
    for i in range(3):
        # Expand the next level Gaussian image
        GE = cv2.pyrUp(gaussian_pyramid[i + 1])
        # Resize to match the current level's size
        GE = cv2.resize(GE, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        # Subtract to get the Laplacian
        L = cv2.subtract(gaussian_pyramid[i], GE)
        laplacian_pyramid.append(L)
    
    return gaussian_pyramid, laplacian_pyramid

def reconstruct_images(gauss1, lapl1, gauss2, lapl2):
    """
    Reconstructs two images using their own Gaussian pyramids and the other image's Laplacian pyramid.
    
    Parameters:
        gauss1 (list): Gaussian pyramid of the first image.
        lapl1 (list): Laplacian pyramid of the first image.
        gauss2 (list): Gaussian pyramid of the second image.
        lapl2 (list): Laplacian pyramid of the second image.
        
    Returns:
        image1 (numpy.ndarray): Reconstructed image using gauss1 and lapl2.
        image2 (numpy.ndarray): Reconstructed image using gauss2 and lapl1.
    """
    # Reconstruct the first image using gauss1 and lapl2
    image1 = gauss1[-1]
    for i in range(2, -1, -1):
        # Upsample the image
        image1 = cv2.pyrUp(image1)
        # Resize to match the Laplacian level's size
        image1 = cv2.resize(image1, (lapl2[i].shape[1], lapl2[i].shape[0]))
        # Add the Laplacian from the other image
        image1 = cv2.add(image1, lapl2[i])
    
    # Reconstruct the second image using gauss2 and lapl1
    image2 = gauss2[-1]
    for i in range(2, -1, -1):
        image2 = cv2.pyrUp(image2)
        image2 = cv2.resize(image2, (lapl1[i].shape[1], lapl1[i].shape[0]))
        image2 = cv2.add(image2, lapl1[i])
    
    return image1, image2


# Read two images
image1 = cv2.imread('Class_Q2/img6.jpg')
image2 = cv2.imread('Class_Q2/img7.jpg')

# Ensure both images are the same size
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Generate pyramids for both images
gauss1, lapl1 = get_pyramids(image1)
gauss2, lapl2 = get_pyramids(image2)

# Reconstruct images by swapping Laplacian pyramids
reconstructed_image1, reconstructed_image2 = reconstruct_images(gauss1, lapl1, gauss2, lapl2)

# Display or save the results
cv2.imshow('Reconstructed Image 1', reconstructed_image1)
cv2.imshow('Reconstructed Image 2', reconstructed_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
