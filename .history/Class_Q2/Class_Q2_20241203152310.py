import cv2
import numpy as np

def generate_pyramids(image):
    # Generate Gaussian Pyramid
    gaussian_pyramid = [image]
    for i in range(2):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    # Generate Laplacian Pyramid
    laplacian_pyramid = []
    for i in range(2, 0, -1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
        laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # The smallest level of Gaussian pyramid is the last level of Laplacian pyramid
    
    return gaussian_pyramid, laplacian_pyramid

def reconstruct_images(gaussian_pyramid1, laplacian_pyramid1, gaussian_pyramid2, laplacian_pyramid2):
    # Reconstruct first image using its Gaussian pyramid and the second image's Laplacian pyramid
    image1 = gaussian_pyramid1[-1]
    for i in range(2, -1, -1):
        image1 = cv2.pyrUp(image1)
        image1 = cv2.add(image1, laplacian_pyramid2[i])
    
    # Reconstruct second image using its Gaussian pyramid and the first image's Laplacian pyramid
    image2 = gaussian_pyramid2[-1]
    for i in range(2, -1, -1):
        image2 = cv2.pyrUp(image2)
        image2 = cv2.add(image2, laplacian_pyramid1[i])
    
    return image1, image2


image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')
gaussian_pyramid1, laplacian_pyramid1 = generate_pyramids(image1)
gaussian_pyramid2, laplacian_pyramid2 = generate_pyramids(image2)
reconstructed_image1, reconstructed_image2 = reconstruct_images(gaussian_pyramid1, laplacian_pyramid1, gaussian_pyramid2, laplacian_pyramid2)
cv2.imwrite('reconstructed_image1.jpg', reconstructed_image1)
cv2.imwrite('reconstructed_image2.jpg', reconstructed_image2)