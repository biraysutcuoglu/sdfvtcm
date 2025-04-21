'''Creates dataset then separates to train, val and test datasets'''
import os
import shutil
import random
import cv2
from src.image_preprocessor_utils import ImageUtils

class DatasetPreprocessor:
    
    @staticmethod
    def move_images_with_masks(image_folder, mask_folder, output_folder_images, output_folder_masks, mask_suffix="_5"):
        num_images_with_masks = 0
        num_images_without_masks = 0

        if not os.path.exists(output_folder_images):
            os.makedirs(output_folder_images)
        if not os.path.exists(output_folder_masks):
            os.makedirs(output_folder_masks)
        
        image_files = os.listdir(image_folder)
        # print(image_files)
        image_extensions = (".png", ".bmp", ".jpg", ".jpeg")
        
        filtered_files = [f for f in image_files if f.lower().endswith(image_extensions)]

        for image_file in filtered_files:
            # construct corresponding mask filename(same name different or same extensions)
            mask_filename = os.path.splitext(image_file)[0] + mask_suffix + ".png"
            # print(mask_filename)

            # Check if the mask exist in the mask folder
            mask_path = os.path.join(mask_folder, mask_filename)
            if os.path.isfile(mask_path):
                # if mask exist copy the image to the output folder
                image_path = os.path.join(image_folder, image_file)
                shutil.copy(image_path, os.path.join(output_folder_images, image_file))
                shutil.copy(mask_path, os.path.join(output_folder_masks, mask_filename))
                num_images_with_masks += 1
            else:
                num_images_without_masks += 1
        
        print(f"Out of {num_images_with_masks + num_images_without_masks} images, {num_images_with_masks} have masks.")
        print(f"These images and masks have been saved to {output_folder_images} and {output_folder_masks}")

    @staticmethod
    def split_dataset(image_folder, mask_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        #Split dataset into train, val and test
        #Move images and masks to respective folders
        
        assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

        # Create output directories
        train_dir = os.path.join(output_folder, "train")
        val_dir = os.path.join(output_folder, "val")
        test_dir = os.path.join(output_folder, "test")

        for folder in [train_dir, val_dir, test_dir]:
            os.makedirs(folder, exist_ok=True)
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(folder, "masks"), exist_ok=True)

        # Get list of all image files
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        images = [f for f in os.listdir(image_folder) if f.lower().endswith(tuple(image_extensions))]

        # Shuffle images randomly
        random.shuffle(images)

        # Split indices
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # Function to move images and masks
        def move_files(image_list, target_dir):
            for img in image_list:
                img_path = os.path.join(image_folder, img)
                mask_path = os.path.join(mask_folder, os.path.splitext(img)[0] + "_5.png")

                shutil.copy(img_path, os.path.join(target_dir, "images", img))
                if os.path.isfile(mask_path):
                    shutil.copy(mask_path, os.path.join(target_dir, "masks", os.path.basename(mask_path)))

        # Move files to respective folders
        move_files(train_images, train_dir)
        move_files(val_images, val_dir)
        move_files(test_images, test_dir)

        print(f"Dataset split completed: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test.")

    # This method was used to resize all images and masks in the dataset to a target size 
    # However, training segmentation model with resized images decreased the performance
    # Therefore, resizing is moved to image_processing_utils.py class 
    # to resize images just before the stable diffusion training
    # @staticmethod
    # def resize_all_images(input_folder, output_folder, target_size=(1024, 1024)):
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
        
    #     image_files = os.listdir(input_folder)
    #     image_extensions = (".png", ".bmp", ".jpg", ".jpeg")
        
    #     filtered_files = [f for f in image_files if f.lower().endswith(image_extensions)]

    #     for image_file in filtered_files:
    #         img_path = os.path.join(input_folder, image_file)
    #         im = cv2.imread(img_path)

    #         if im is None:
    #             print(f"Error reading image {img_path}, skipping...")
    #             continue

    #         resized_im, _, _ = ImageUtils.letterbox(im, target_size)
    #         output_path = os.path.join(output_folder, image_file)
    #         cv2.imwrite(output_path, resized_im)
        
    #     print(f"Resized {len(filtered_files)} images to {target_size} and saved to {output_folder}")


