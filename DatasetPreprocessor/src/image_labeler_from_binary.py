from PIL import Image
import numpy as np
import os
from collections import Counter
import json

'''
This class calculates the amount of wear along y axis of the tool and generates a label for each image from the binary masks.
It is implemented for labelling Tool Dataset images to be used for training the Stable Diffusion model.
'''
class ImageLabeler:
    
    def measure_wear(mask_path):
        '''
        Measures the amount of wear in the image by counting the number of 
        continuos white pixels along y axis. in the binary mask.
        Args:
            mask_path (str): Path to the binary mask image.
        Returns:
            int: The amount of wear in the image.
        '''
        img = Image.open(mask_path).convert("L")  
        img_arr = np.array(img)

        # Convert to binary image (255 for white, 0 for black)
        binary_img = (img_arr > 127).astype(np.uint8)  # Convert to 0 (black) or 1 (white)

        # Find the max continuous white pixels in Y-direction
        # Time efficient way to find the max continuous white pixels (rather than using a nested loop)
        max_wear = np.max(np.apply_along_axis(lambda col: np.max(np.diff(np.where(np.concatenate(([0], col, [0])) == 0)[0]) - 1), axis=0, arr=binary_img))

        return max_wear

    def measure_wear_all_images(mask_dir):
        # Iterate through all images in the directory
        wear_dict = {}
        for mask_path in os.listdir(mask_dir):
            if mask_path.endswith('.png'):
                wear_value = ImageLabeler.measure_wear(os.path.join(mask_dir, mask_path))
                image_filename = mask_path.replace("_mask", "")
                wear_dict[image_filename] = wear_value
        return wear_dict

    
    #all images are approximately taken from same distance and same tool
    #so we can determine the amount of wear by taking only the number of wear pixels
    def assign_categories(wear_dict):
        ''' 
        Assigns categories based on wear values 
        Args:
            wear_dict (dict): Dictionary with image filenames as keys and wear values as values.
        Returns:
            dict: Dictionary with image filenames as keys and wear categories as values.
        '''
        wear_categories = {}
        for filename, wear in wear_dict.items():
            if wear <= 100:
                wear_categories[filename] = "Low Wear"
            elif 101 <= wear <= 250:
                wear_categories[filename] = "Moderate Wear"
            elif wear > 250:
                wear_categories[filename] = "High Wear"

        return wear_categories  # Return the categorized dictionary
    
    def display_wear_info(wear_dict, categorized_wear):
        ''' Prints wear information'''
        print(f"Number of images: {len(wear_dict)}")

        max_file = max(wear_dict, key=wear_dict.get)
        min_file = min(wear_dict, key=wear_dict.get)

        print(f"Max wear value: {max(wear_dict.values())} in file {max_file}")
        print(f"Min wear value: {min(wear_dict.values())} in file {min_file}")

        # ------------------
        # display number of images in each wear category (low, moderate, high)
        category_counts = Counter(categorized_wear.values())
        print()
        print("Number of images in each category: ")
        for category, count in category_counts.items():
            print(f"{category}: {count}")
    
    def generate_image_prompts(categorized_wear, output_dir):
        output_file = os.path.join(output_dir, "metadata.jsonl")
        num_labels = 0
        with open(output_file, "w") as f:
            for k, v in categorized_wear.items():
                file_name = k
        
                prompt = f"Insert with {v}"

                data = {"file_name": file_name, "text": prompt}
                f.write(json.dumps(data) + "\n")
                num_labels += 1
        
        print(f"Labels of {num_labels} images, written to {output_file}")




    
    



    

       