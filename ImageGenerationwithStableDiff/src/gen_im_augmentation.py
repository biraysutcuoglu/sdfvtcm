from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
import random

'''
This class measures brightness and contrast of synthetic and real images
adjusts the synthetic images according to real images
'''

class GeneratedImageAugmentation:
    def measure_brightness(image_path):
        image = Image.open(image_path)

        # Convert the image to grayscale
        gray_image = image.convert('L')
        gray_image_np = np.array(gray_image)

        brightness = np.mean(gray_image_np)
        return brightness
    
    # root mean square contrast
    def measure_rms_contrast(image_path):
        im = Image.open(image_path)
        
        # Convert grayscale
        gray_image = im.convert('L')
        im_np = np.array(gray_image)

        mean_intensity = np.mean(im_np)
        squared_diff = (im_np - mean_intensity)**2

        mean_squared_diff = np.mean(squared_diff)
        
        rms_contrast_val = np.sqrt(mean_squared_diff)
        return rms_contrast_val
    
    def measure_brightness_of_dataset(image_dir):
        brightness_values = []

        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            # Ensure it's a valid image file
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                brightness = GeneratedImageAugmentation.measure_brightness(file_path)
                brightness_values.append(brightness)
        
        average_brightness = np.mean(brightness_values)
        max_brightness = np.max(brightness_values)
        min_brightness = np.min(brightness_values)
        brightness_variation = (max_brightness - min_brightness) / average_brightness * 100
        
        print("Min brightness: ", min_brightness, "Max brightness: ", max_brightness, )
        print("Average brightness: ", average_brightness)
        print("Brightness variation %", brightness_variation)
        
        return brightness_variation

    def measure_contrast_of_dataset(image_dir):
        contrast_values = []

        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            # Ensure it's a valid image file
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                contrast = GeneratedImageAugmentation.measure_rms_contrast(file_path)
                contrast_values.append(contrast)
        
        print("Min rms contrast: ", np.min(contrast_values), "Max rms contrast: ", np.max(contrast_values), )
        print("Average rms contrast: ", np.mean(contrast_values))

        # Compute percentage change
        percentage_change = ((np.max(contrast_values) - np.min(contrast_values)) / np.max(contrast_values)) * 100
        print("Contrast change %", percentage_change)

    def adjust_brightness(image_path, real_brightness_var, generated_brightness_var, visualize="no"):
        image = Image.open(image_path)
        
        variation_factor = real_brightness_var / generated_brightness_var
    
        min_factor = max(0.5, 1 - (variation_factor / 10))  # Ensure it doesn't go below 0.5
        max_factor = min(1.5, 1 + (variation_factor / 10))  # Ensure it doesn't exceed 1.5
        
        brightness_factor = random.uniform(min_factor, max_factor)

        enhancer = ImageEnhance.Brightness(image)
        aug_image = enhancer.enhance(brightness_factor)

        if visualize == "yes":
            GeneratedImageAugmentation.visualize_images(image, aug_image, "brightness", brightness_factor)
        
        return aug_image
    
    def adjust_contrast(image_path, visualize="no"):
        im = Image.open(image_path)
        
        # generate random contrast 
        contrast_factor = random.uniform(1.0, 1.4)

        # enhance contrast
        enhancer = ImageEnhance.Contrast(im)
        image_enhanced = enhancer.enhance(contrast_factor)

        if visualize == "yes":
            GeneratedImageAugmentation.visualize_images(im, image_enhanced, "contrast", contrast_factor)
        
        return image_enhanced

    def adjust_brightness_of_images(image_dir, output_dir, real_brightness_var, generated_brightness_var):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                brightness_adj_im = GeneratedImageAugmentation.adjust_brightness(file_path, real_brightness_var, generated_brightness_var)

                output_path = os.path.join(output_dir, filename)
                brightness_adj_im.save(output_path)
        
        print("Images are saved to ", output_dir)

    def adjust_contrast_of_images(image_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                contrast_adj_im = GeneratedImageAugmentation.adjust_contrast(file_path)

                output_path = os.path.join(output_dir, filename)
                contrast_adj_im.save(output_path)
        
        print("Images are saved to ", output_dir)


    def visualize_images(im_1, im_2, aug_type, aug_factor):
        plt.figure(figsize=(10, 5))  # Ensure proper figure size

        # Plot the original image 
        plt.subplot(1, 2, 1) 
        plt.title("Original") 
        plt.imshow(im_1) 
        plt.axis("off")

        # Plot the brightness-adjusted image
        plt.subplot(1, 2, 2) 
        if(aug_type == "brightness"):
            plt.title(f"Brightness by factor {aug_factor:.2f}") 
        elif (aug_type == "contrast"):
            plt.title((f"Contrast by factor {aug_factor:.2f}") )
        
        plt.imshow(im_2) 
        plt.axis("off")

        plt.show()  



    
        