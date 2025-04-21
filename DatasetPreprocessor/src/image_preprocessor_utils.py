from PIL import Image
from os import listdir
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""This class is for resizing images for the Stable Diff Model because the model works with 1024x1024 images
Also enhances the images by increasing the contrast"""

class ImageUtils:
    @staticmethod
    def resize_with_Lanczos(input_image_path, target_image_width, target_image_height):
        img = Image.open(input_image_path)
        return img.resize((target_image_width, target_image_height), Image.LANCZOS)
    
    @staticmethod
    def resize_with_Bicubic(input_image_path, target_image_width, target_image_height):
        img = Image.open(input_image_path)
        return img.resize((target_image_width, target_image_height), Image.BICUBIC)

    @staticmethod
    def resize_all_images(input_dir, output_dir, resizing_alg, target_image_width, target_image_height, ratio=None, padding=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        num_resized_images = 0
        for image_path in os.listdir(input_dir):
            if (image_path.lower().endswith(".png") or image_path.lower().endswith(".bmp")):
                input_path = os.path.join(input_dir, image_path)

                resized_image_name = f"{image_path}"
                output_path = os.path.join(output_dir, resized_image_name)

                if resizing_alg == "lanczos":
                    resized_img = ImageUtils.resize_with_Lanczos(input_path, target_image_width, target_image_height)
                    num_resized_images += 1
                elif resizing_alg == "bicubic":
                    resized_img = ImageUtils.resize_with_Bicubic(input_path, target_image_width, target_image_height)
                    num_resized_images += 1
                elif resizing_alg == "letterbox":
                    resized_img_np, _, _ = ImageUtils.letterbox(input_path, new_shape=(target_image_width, target_image_height))
                    resized_img = Image.fromarray(resized_img_np)
                    num_resized_images += 1
                elif resizing_alg == "undo_letterbox":
                    resized_img = ImageUtils.undo_letterbox(input_path, (target_image_width, target_image_height), ratio, padding)
                    num_resized_images += 1
                else:
                    print(f"Invalid resizing algorithm {resizing_alg}")
                    return
            
                resized_img.save(output_path)
        
        print(f"{num_resized_images} images are resized to {target_image_width}x{target_image_height} using {resizing_alg} and saved to {output_dir}")
    
    @staticmethod
    def letterbox(image_path, new_shape=(1024, 1024), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Implementation from official YOLO5 repository
        # Resizes the image to the target size with letterbox padding (keeping the aspect ratio)
        
        # Resize and pad image while meeting stride-multiple constraints
        image_pil = Image.open(image_path)
        im = np.array(image_pil)
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    @staticmethod
    def undo_letterbox(image_path, original_shape, ratio, padding):
        """
        Undo the letterbox padding and resize the image to the original shape
        """
        # Read image
        image_pil = Image.open(image_path)
        image = np.array(image_pil)
        
        # Unpad
        dw, dh = padding
        top, bottom = int(dh), int(dh)
        left, right = int(dw), int(dw)

        # Remove padding by cropping
        unpadded_image = image[top:image.shape[0] - bottom, left:image.shape[1] - right]

        # Resize back to original shape
        restored_image = cv2.resize(unpadded_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

        return restored_image

    @staticmethod
    def copy_metadata_file_to_resized_images_dir(metadata_dir, resized_images_dir, metadata_filename="metadata.jsonl"):
        # Search for the metadata file in the given directory
        metadata_file_path = None
        for root, _, files in os.walk(metadata_dir):
            if metadata_filename in files:
                metadata_file_path = os.path.join(root, metadata_filename)
                break

        # If file is found, copy it to the target directory
        if metadata_file_path:
            output_path = os.path.join(resized_images_dir, metadata_filename)
            shutil.copy(metadata_file_path, output_path)
            print(f"Metadata file {metadata_filename} copied to {resized_images_dir}")
        else:
            print(f"Metadata file {metadata_filename} not found in {metadata_dir}")
    
    @staticmethod
    def apply_clahe(image_dir, output_dir):
        # Apply histogram equalization to enhance the contrast of the images
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_clahe_images = 0
        for image_path in os.listdir(image_dir):
            if (image_path.lower().endswith(".png") or image_path.lower().endswith(".bmp")):
                input_path = os.path.join(image_dir, image_path)

                enhanced_image_name = f"{image_path}"
                output_path = os.path.join(output_dir, enhanced_image_name)

                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_img = clahe.apply(img)
            
                cv2.imwrite(output_path, enhanced_img)
                num_clahe_images += 1
        
        print(f"{num_clahe_images} images are enhanced with CLAHE and saved to {output_dir}.")

    @staticmethod
    def png_to_bmp_all_images(png_image_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        num_type_changed_images = 0
        for image_path in os.listdir(png_image_dir):
            if image_path.lower().endswith(".png"):
                input_path = os.path.join(png_image_dir, image_path)

                type_changed_file = os.path.splitext(image_path)[0] + ".bmp"
                output_path = os.path.join(output_dir, type_changed_file)

                try:
                    # open and convert to bmp
                    with Image.open(input_path) as img:
                        img.save(output_path, format="BMP")
                    
                    num_type_changed_images += 1
                except:
                    print(f"Failed to convert to .bmp format {image_path}")
        
        print(f"Conversion completed. {num_type_changed_images} files converted to .bmp format.")


    @staticmethod
    def show_enhanced_image(img, enhanced_image):
        # Show real and clahe image side by side
        # Plot the images side by side
        plt.figure(figsize=(10,5))

        # Original Image
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # CLAHE Image
        plt.subplot(1,2,2)
        plt.imshow(enhanced_image, cmap='gray')
        plt.title('CLAHE Image')
        plt.axis('off')

        # Show the images
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_og_and_resized_image(og_img, resized_img):
        # Show original and resized image side by side
        # Plot the images side by side
        plt.figure(figsize=(5,10))

        # Original Image
        plt.subplot(1,2,1)
        plt.imshow(og_img)
        plt.title('Original Image')
        plt.axis('off')

        # Resized Image
        plt.subplot(1,2,2)
        plt.imshow(resized_img)
        plt.title('Resized Image')
        plt.axis('off')

        # Show the images
        plt.tight_layout()
        plt.show()







                






